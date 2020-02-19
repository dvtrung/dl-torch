"""Train a model."""
import os
import random
import sys
import traceback
from datetime import datetime
from typing import Tuple, Dict, List

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from dlex import FrameworkBackend
from dlex.configs import Configs
from dlex.datasets.torch import Dataset
from dlex.datatypes import ModelReport
from dlex.torch.models.base import DataParellelModel, BaseModel
from dlex.torch.utils.model_utils import get_model
from dlex.utils import set_seed, check_interval_passed, Datasets, table2str
from dlex.utils.logging import logger, epoch_info_logger, log_result, json_dumps, \
    log_outputs
from dlex.utils.model_utils import get_dataset

DEBUG_NUM_ITERATIONS = 5
DEBUG_BATCH_SIZE = 4


class PytorchBackend(FrameworkBackend):
    def __init__(self, argv=None, params=None, configs: Configs = None, training_idx: int = None, report_queue=None):
        super().__init__(argv, params, configs, training_idx, report_queue)
        """Read config and train model."""
        logger.info(f"Training started ({training_idx}).")

        self.report.metrics = params.test.metrics
        self.report.results = {m: None for m in self.report.metrics}

    def run_train(self):
        train_cfg = self.params.train
        report = self.report

        if train_cfg.cross_validation:
            set_seed(self.params.random_seed)
            results = []
            for i in range(train_cfg.cross_validation):
                summary_writer = SummaryWriter(os.path.join(self.configs.log_dir, "runs", str(self.training_idx), str(i + 1)))
                self.report.cv_current_fold = i + 1
                self.report.cv_num_folds = train_cfg.cross_validation
                self.params.dataset.cv_current_fold = i + 1
                self.params.dataset.cross_validation = train_cfg.cross_validation
                self.report_queue.put(self.report)

                model, datasets = self.load_model("train")

                res = self.train(
                    model, datasets, summary_writer=summary_writer,
                    tqdm_desc=f"[{self.training_idx}] CV {self.report.cv_current_fold}/{self.report.cv_num_folds} - ",
                    tqdm_position=self.training_idx)
                results.append(res)
                report.results = {
                    metric: [r[metric] for r in results] for metric in results[0]}
                self.report_queue.put(report)
                summary_writer.close()

            logger.info(f"Training finished. Results: {str(report.results)}")
            report.finish()
        else:
            summary_writer = SummaryWriter(os.path.join(self.params.log_dir, "runs", str(self.training_idx)))
            set_seed(self.params.random_seed)
            model, datasets = self.load_model("train")
            res = self.train(
                model, datasets, summary_writer,
                tqdm_position=self.training_idx,
                on_epoch_finished=lambda r: self.report_queue.put(r))
            report.results = res
            report.finish()
            summary_writer.close()

        return self.report

    def run_evaluate(self):
        model, datasets = self.load_model("test")

        for mode in self.params.train.eval:
            result, outputs = self.evaluate(
                model, datasets.get_dataset(mode),
                output_path=os.path.join(self.params.log_dir, "results", f"{self.args.load}_{mode}"))
            # for output in random.choices(outputs, k=50):
            #     logger.info(str(output))
            logger.info(str(result))

    def load_model(self, mode):
        """
        Load model and dataset
        :param mode: train, test, dev
        :return:
        """
        params = self.params

        if not self.configs:
            self.configs = Configs(mode=mode, argv=self.argv)
            envs, args = self.configs.environments, self.configs.args
            assert len(envs) == 1
            assert len(envs[0].configs_list) == 1
            params = envs[0].configs_list[0]
        else:
            args = self.configs.args

        self.report.metrics = params.test.metrics

        if mode == "train":
            if args.debug:
                params.train.batch_size = DEBUG_BATCH_SIZE
                params.test.batch_size = DEBUG_BATCH_SIZE

        # Init dataset
        dataset_builder = get_dataset(params)
        assert dataset_builder, "Dataset not found."
        if not args.no_prepare:
            dataset_builder.prepare(download=args.download, preprocess=args.preprocess)
        if mode == "test":
            datasets = Datasets("pytorch")
            for mode in params.train.eval:
                datasets.load_dataset(dataset_builder, mode)
        elif mode == "train":
            if args.debug:
                datasets = Datasets(
                    "pytorch",
                    train=dataset_builder.get_pytorch_wrapper("test"),
                    test=dataset_builder.get_pytorch_wrapper("test"))
            else:
                datasets = Datasets(
                    "pytorch",
                    train=dataset_builder.get_pytorch_wrapper("train"),
                    valid=dataset_builder.get_pytorch_wrapper("valid") if "valid" in params.train.eval else
                    dataset_builder.get_pytorch_wrapper("dev") if "dev" in params.train.eval else
                    None,
                    test=dataset_builder.get_pytorch_wrapper("test") if "test" in params.train.eval else None)
        else:
            raise ValueError("Mode '%s' is not valid" % mode)

        # Init model
        model_cls = get_model(params)
        assert model_cls, "Model not found."
        model = model_cls(params, datasets.train if datasets.train is not None else datasets.test or datasets.valid)
        # model.summary()

        # log model summary
        parameter_details = [["Name", "Shape", "Trainable"]]
        num_params = 0
        num_trainable_params = 0
        for name, parameter in model.named_parameters():
            parameter_details.append([
                name,
                str(list(parameter.shape)),
                "✓" if parameter.requires_grad else ""])
            num_params += np.prod(list(parameter.shape))
            if parameter.requires_grad:
                num_trainable_params += np.prod(list(parameter.shape))

        s = table2str(parameter_details)
        logger.debug(f"Model parameters\n{s}")
        logger.debug(" - ".join([
            f"No. parameters: {num_params:,}",
            f"No. trainable parameters: {num_trainable_params:,}"
        ]))
        self.report.param_details = s
        self.report.num_params = num_params
        self.report.num_trainable_params = num_trainable_params

        use_cuda = torch.cuda.is_available()
        if use_cuda and params.gpu:
            gpus = [f"cuda:{g}" for g in params.gpu]
            model = DataParellelModel(model, gpus)
            logger.info("Start training using %d GPU(s): %s", len(params.gpu), str(params.gpu))
            torch.cuda.set_device(torch.device(gpus[0]))
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(gpus[0])
        else:
            model = DataParellelModel(model, ['cpu'])

        logger.debug("Dataset: %s. Model: %s", str(dataset_builder), str(model_cls))
        if use_cuda:
            logger.info("CUDA available: %s", torch.cuda.get_device_name(0))

        # Load checkpoint or initialize new training
        if args.load:
            self.configs.training_id = model.load_checkpoint(args.load)
            logger.info("Loaded checkpoint: %s", args.load)
            if mode == "train":
                logger.info("EPOCH: %f", model.global_step / len(datasets.train))

        return model, datasets

    def train(
            self,
            model: DataParellelModel,
            datasets: Datasets,
            summary_writer: SummaryWriter,
            tqdm_desc="",
            tqdm_position=None,
            on_epoch_finished=None):
        """
        :param model:
        :param datasets:
        :param summary_writer:
        :param tqdm_desc:
        :param tqdm_position:
        :param on_epoch_finished:
        :return:
        """
        report = self.report
        args = self.configs.args
        train_cfg = self.params.train
        params = self.params

        epoch = model.global_step // len(datasets.train)
        num_samples = model.global_step % len(datasets.train)
        report.current_results = {}

        report.num_epochs = train_cfg.num_epochs
        report.epoch_losses = []
        report.epoch_valid_results = []
        report.epoch_test_results = []
        report.current_results = {}

        # num_samples = 0
        for current_epoch in range(epoch + 1, train_cfg.num_epochs + 1):
            log_dict = dict(epoch=current_epoch)
            log_dict['total_time'], loss = self.train_epoch(
                current_epoch, model, datasets, report, num_samples,
                tqdm_desc=tqdm_desc + f"Epoch {current_epoch}",
                tqdm_position=tqdm_position)
            report.epoch_losses.append(loss)
            summary_writer.add_scalar(f"loss", loss, current_epoch)
            log_dict['loss'] = loss
            num_samples = 0

            def _evaluate(mode):
                # Evaluate model
                result, outputs = self.evaluate(
                    model, getattr(datasets, mode),
                    output_path=os.path.join(params.log_dir, "results", "latest"),
                    tqdm_desc=tqdm_desc + f"Epoch {current_epoch}",
                    tqdm_position=None if tqdm_position is None else tqdm_position)
                best_result = log_result(mode, params, result, datasets.train.builder.is_better_result)
                # for metric in best_result:
                #     if best_result[metric] == result:
                #         model.save_checkpoint(
                #             "best" if len(params.test.metrics) == 1 else "%s-best-%s" % (mode, metric))
                #         logger.info("Best %s for %s set reached: %f", metric, mode, result['result'][metric])
                return result, best_result, outputs

            if datasets.test is not None:
                test_result, test_best_result, test_outputs = _evaluate("test")
                report.epoch_test_results.append(test_result['result'])
                log_outputs("test", params, test_outputs)
                log_dict['test_result'] = test_result['result']
                for metric in test_result['result']:
                    summary_writer.add_scalar(
                        f"test_{metric}",
                        test_result['result'][metric], current_epoch)

            if datasets.valid is not None:
                valid_result, valid_best_result, valid_outputs = _evaluate("valid")
                report.epoch_valid_results.append(valid_result['result'])
                log_outputs("valid", params, valid_outputs)
                log_dict['valid_result'] = valid_result['result']
                for metric in valid_result['result']:
                    summary_writer.add_scalar(
                        f"valid_{metric}",
                        valid_result['result'][metric], current_epoch)

            # results for reporting
            if train_cfg.select_model == "last":
                report.current_results = test_result['result']
            elif train_cfg.select_model == "best":
                if datasets.valid is None:
                    # there's no valid set, report test result with lowest loss
                    if loss <= min(report.epoch_losses):
                        report.current_results = test_result['result']
                        logger.info("Result updated (lowest loss reached: %.4f) - %s" % (
                            loss,
                            ", ".join(["%s: %.2f" % (metric, res) for metric, res in report.current_results.items()])
                        ))
                else:
                    for metric in valid_best_result:
                        if valid_best_result[metric] == valid_result:
                            if datasets.test is not None:
                                # report test result of best model on valid set
                                logger.info("Best result: %f", test_result['result'][metric])
                                report.current_results[metric] = test_result['result'][metric]
                                log_result(f"valid_test_{metric}", params, test_result, datasets.train.builder.is_better_result)
                                log_outputs("valid_test", params, test_outputs)
                            else:
                                # there's no test set, report best valid result
                                log_result(f"valid_{metric}", params, valid_result, datasets.train.builder.is_better_result)
                                report.current_results[metric] = valid_result['result'][metric]
                                log_outputs("valid", params, valid_outputs)

            if args.output_test_samples:
                logger.info("Random samples")
                for output in random.choices(test_outputs if datasets.test is not None else valid_outputs, k=5):
                    logger.info(str(output))

            epoch_info_logger.info(json_dumps(log_dict))
            log_msgs = [
                "time: %s" % log_dict['total_time'].split('.')[0],
                "loss: %.4f" % log_dict['loss']
            ]

            for metric in report.metrics:
                if datasets.valid:
                    log_msgs.append(f"dev ({metric}): %.2f" % (
                        log_dict['valid_result'][metric],
                        # valid_best_result[metric]['result'][metric]
                    ))
                if datasets.test:
                    log_msgs.append(f"test ({metric}): %.2f" % (
                        log_dict['test_result'][metric],
                        # test_best_result[metric]['result'][metric],
                    ))
            logger.info(f"session {report.training_idx} - epoch {current_epoch}: " + " - ".join(log_msgs))

            # Early stopping
            if params.train.early_stop:
                ne = params.train.early_stop.num_epochs
                min_diff = params.train.early_stop.min_diff or 0.
                if datasets.valid is not None:
                    last_results = report.epoch_valid_results
                    if len(last_results) > ne:
                        if all(
                                max([r[metric] for r in last_results[-ne:]]) <=
                                max([r[metric] for r in last_results[:-ne]])
                                for metric in report.metrics):
                            logger.info("Early stop at epoch %s", current_epoch)
                            break
                else:
                    losses = report.epoch_losses
                    if len(losses) > ne:
                        diff = min(losses[:-ne]) - min(losses[-ne:])
                        logger.debug("Last %d epochs decrease: %.4f", ne, diff)
                        if diff <= min_diff:
                            logger.info("Early stop at epoch %s", current_epoch)
                            break

            if on_epoch_finished:
                on_epoch_finished(report)

        return report.current_results

    def train_epoch(
            self,
            current_epoch: int,
            model: DataParellelModel,
            datasets: Datasets,
            report: ModelReport,
            num_samples=0,
            tqdm_desc="Epoch {current_epoch}",
            tqdm_position=None):
        """Train."""
        report.current_epoch = current_epoch
        args = self.configs.args
        params = self.params

        if self.params.dataset.shuffle:
            datasets.train.shuffle()

        model.start_calculating_loss()
        start_time = datetime.now()

        if isinstance(params.train.batch_size, int):  # fixed batch size
            batch_sizes = {0: params.train.batch_size}
        elif isinstance(params.train.batch_size, dict):
            batch_sizes = params.train.batch_size
        else:
            raise ValueError("Batch size is not valid.")

        for key in batch_sizes:
            batch_sizes[key] *= (len(self.params.gpu) if self.params.gpu else 1) or 1
        assert 0 in batch_sizes

        total = len(datasets.train)
        last_save = 0
        last_log = 0
        with tqdm(
                desc=tqdm_desc.format(current_epoch=current_epoch),
                total=total, leave=False,
                position=tqdm_position,
                disable=not args.show_progress) as t:
            t.update(num_samples)
            batch_size_checkpoints = sorted(batch_sizes.keys())
            for start, end in zip(batch_size_checkpoints, batch_size_checkpoints[1:] + [100]):
                if end / 100 < num_samples / len(datasets.train):
                    continue
                batch_size = batch_sizes[start]
                data_train = datasets.train.get_iter(
                    batch_size,
                    start=max(start * len(datasets.train) // 100, num_samples),
                    end=end * len(datasets.train) // 100
                )

                for epoch_step, batch in enumerate(data_train):
                    loss = model.training_step(batch)
                    try:
                        if batch is None or len(batch) == 0:
                            raise Exception("Batch size 0")
                        # loss = model.training_step(batch)
                        # clean
                        torch.cuda.empty_cache()
                    except RuntimeError as e:
                        torch.cuda.empty_cache()
                        logger.error(str(e))
                        logger.info("Saving model before exiting...")
                        model.save_checkpoint("latest")
                        sys.exit(2)
                    except Exception as e:
                        logger.error(str(e))
                        continue
                    else:
                        t.set_postfix(
                            # loss="%.4f" % loss,
                            loss="%.4f" % model.epoch_loss,
                            # lr=mean(model.learning_rates())
                            **(report.current_results or {})
                        )

                    # if args.debug and epoch_step > DEBUG_NUM_ITERATIONS:
                    #    break
                    t.update(batch_size)
                    num_samples += batch_size
                    progress = 1. if total - num_samples < batch_size else num_samples / total

                    model.current_epoch = current_epoch
                    model.global_step = (current_epoch - 1) * len(datasets.train) + num_samples

                    if report.summary_writer is not None:
                        report.summary_writer.add_scalar("loss", loss, model.global_step)

                    # Save model
                    is_passed, last_save = check_interval_passed(last_save, params.train.save_every, progress)
                    if is_passed:
                        if args.save_all:
                            model.save_checkpoint("epoch-%02d" % current_epoch)
                        else:
                            model.save_checkpoint("latest")

                    # Log
                    is_passed, last_log = check_interval_passed(last_log, params.train.log_every, progress)
                    if is_passed:
                        logger.info(", ".join([
                            f"epoch: {current_epoch}",
                            f"progress: {int(progress * 100)}%",
                            f"epoch_loss: {model.epoch_loss:.4f}",
                        ]))

                    if args.debug:
                        input("Press any key to continue...")
                model.end_training_epoch()
        # model.save_checkpoint("epoch-latest")
        end_time = datetime.now()
        return str(end_time - start_time), model.epoch_loss

    def evaluate(
            self,
            model: BaseModel,
            dataset: Dataset,
            output_path,
            tqdm_desc="Eval",
            tqdm_position=None) -> Tuple[Dict, List]:
        """
        Evaluate model and save result.
        :param model:
        :param dataset:
        :param output_path: path without extension
        :param tqdm_desc:
        :param tqdm_position:
        :return:
        """
        params = self.params
        report = self.report

        model.module.eval()
        torch.cuda.empty_cache()
        last_log = 0
        with torch.no_grad():
            data_iter = dataset.get_iter(
                batch_size=params.test.batch_size or params.train.batch_size)

            # total = {key: 0 for key in params.test.metrics}
            # acc = {key: 0. for key in params.test.metrics}
            results = {metric: 0. for metric in params.test.metrics}
            outputs = []
            y_pred_all, y_ref_all, extra_all = [], [], []
            for batch in tqdm(
                    data_iter,
                    desc=tqdm_desc,
                    leave=False,
                    position=tqdm_position,
                    disable=not self.configs.args.show_progress):
                # noinspection PyBroadException
                try:
                    if batch is None or len(batch) == 0:
                        raise Exception("Batch size 0")

                    inference_outputs = model.infer(batch)
                    y_pred, y_ref, others = inference_outputs[0], inference_outputs[1], inference_outputs[2:]
                    y_pred_all += y_pred
                    y_ref_all += y_ref
                    # for metric in params.test.metrics:
                    #     if metric == "loss":
                    #         loss = model.get_loss(batch, model_output).item()
                    #         _acc, _total = loss * len(y_pred), len(y_pred)
                    #     else:
                    #         _acc, _total = dataset.evaluate_batch(y_pred, batch, metric=metric)
                    #     acc[metric] += _acc
                    #     total[metric] += _total

                    for i, predicted in enumerate(y_pred):
                        str_input, str_ground_truth, str_predicted = dataset.format_output(
                            predicted, batch.item(i))
                        outputs.append(dict(
                            input=str_input,
                            reference=str_ground_truth,
                            hypothesis=str_predicted))

                        is_passed, last_log = check_interval_passed(last_log, params.test.log_every)
                        if is_passed:
                            logger.debug(
                                "sample %d\n\t[inp] %s\n\t[ref] %s\n\t[hyp] %s",
                                len(outputs),
                                str(outputs[-1]['input']),
                                str(outputs[-1]['reference']),
                                str(outputs[-1]['hypothesis']))

                    if report.summary_writer is not None:
                        model.write_summary(report.summary_writer, batch, (y_pred, others))
                except Exception:
                    logger.error(traceback.format_exc())

            for metric in params.test.metrics:
                results[metric] = dataset.evaluate(y_pred_all, y_ref_all, metric, output_path)

        result = {
            "epoch": "%.1f" % model.current_epoch,
            "result": {key: results[key] for key in results}
        }

        return result, outputs