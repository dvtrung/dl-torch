"""Train a model."""
import os
import random
import sys
import traceback
from collections import namedtuple, defaultdict
from datetime import datetime
from typing import Callable, Dict

import torch
from dlex import FrameworkBackend, TrainingProgress
from dlex.configs import Configs, Params
from dlex.datasets.torch import Dataset
from dlex.datatypes import ModelReport
from dlex.torch.models.base import BaseModel, ModelWrapper
from dlex.torch.utils.model_utils import get_model
from dlex.utils import check_interval_passed, Datasets
from dlex.utils.logging import logger, epoch_info_logger, log_result, json_dumps, \
    log_outputs
from dlex.utils.model_utils import get_dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

DEBUG_NUM_ITERATIONS = 5
DEBUG_BATCH_SIZE = 4


EvaluationResults = namedtuple("EvaluationResults", "results outputs")


class PytorchBackend(FrameworkBackend):
    def __init__(self, params: Params, training_idx: int = 0, report_queue=None):
        super().__init__(params, training_idx, report_queue)

    def run_cross_validation_training(self) -> ModelReport:
        report = self.report
        report.results = []
        train_cfg = self.params.train
        for i in range(train_cfg.cross_validation):
            # Reset random seed so the same order is returned after shuffling dataset
            self.set_seed()

            summary_writer = SummaryWriter(
                os.path.join(self.configs.log_dir, "runs", str(self.training_idx), str(i + 1)))
            self.params.dataset.cv_current_fold = i + 1
            self.params.dataset.cv_num_folds = train_cfg.cross_validation
            self.update_report()

            model, datasets = self.load_model("train")

            results = self.train(
                model, datasets, summary_writer=summary_writer,
                tqdm_desc=f"[{self.params.env_name}-{self.training_idx}] CV {i + 1}/{train_cfg.cross_validation} - ",
                tqdm_position=self.training_idx)
            report.results.append(results)
            self.update_report()
            summary_writer.close()

        logger.info(f"Training finished.")
        for metric in report.metrics:
            logger.info(f"Results ({metric}): {report.get_result_text()}")
        report.finish()
        return report

    def run_train(self) -> ModelReport:
        logger.info(f"Training started ({self.training_idx})")
        report = self.report
        report.results = {name: {m: None for m in self.report.metrics} for name in self.report.test_sets}

        if self.params.train.cross_validation:
            return self.run_cross_validation_training()
        else:
            summary_writer = SummaryWriter(os.path.join(self.params.log_dir, "runs", str(self.training_idx)))
            model, datasets = self.load_model("train")
            res = self.train(
                model, datasets, summary_writer,
                tqdm_position=self.training_idx,
                on_epoch_finished=self.update_report)
            report.results = res
            report.finish()
            summary_writer.close()

            return report

    def run_evaluate(self):
        model, datasets = self.load_model("test")

        for name, dataset in datasets.test_sets.items():
            logger.info(f"Evaluate model on dataset '{name}'")
            logger.info(f"Log dir: {self.configs.log_dir}")
            ret = self.evaluate(
                model, dataset,
                output_path=os.path.join(self.params.log_dir, "results"),
                output_tag=f"{self.args.load}_{name}")
            # for output in random.choices(outputs, k=50):
            #     logger.info(str(output))

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

        if mode == "train":
            if args.debug:
                params.train.batch_size = DEBUG_BATCH_SIZE
                params.test.batch_size = DEBUG_BATCH_SIZE

        # Init dataset
        dataset_builder = get_dataset(params)
        assert dataset_builder, "Dataset not found."
        if not args.no_prepare:
            dataset_builder.prepare(download=args.download, preprocess=args.preprocess)

        datasets = Datasets(
            "pytorch", dataset_builder,
            train_set=params.train.train_set,
            valid_set=params.train.valid_set,
            test_sets=params.test.test_sets)

        # Init model
        model_cls = get_model(params)
        assert model_cls, "Model not found."
        model = model_cls(params, next(iter(datasets.test_sets.values())) or datasets.train_set or datasets.valid_set)

        # log model summary
        self.report.set_model_summary(
            variable_names=[name for name, _ in model.named_parameters()],
            variable_shapes=[list(p.shape) for _, p in model.named_parameters()],
            variable_trainable=[p.requires_grad for _, p in model.named_parameters()]
        )

        if torch.cuda.is_available() and params.gpu:
            logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
            gpus = [f"cuda:{g}" for g in params.gpu]
            model = ModelWrapper(model, gpus)
            logger.info("Preparing %d GPU(s): %s", len(params.gpu), str(params.gpu))
            torch.cuda.set_device(torch.device(gpus[0]))
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            model = ModelWrapper(model)

        logger.debug("Dataset: %s. Model: %s", str(dataset_builder), str(model_cls))

        # Load checkpoint or initialize new training
        if args.load:
            self.configs.training_id = model.load_checkpoint(args.load)
            logger.info("Loaded checkpoint: %s", args.load)
            if mode == "train":
                logger.info("EPOCH: %f", model.global_step / len(datasets.train_set))

        return model, datasets

    def record_results(
            self,
            select_model: str,
            model,
            datasets) -> bool:
        """

        :param select_model:
        :return: whether the results are updated
        """
        report = self.report
        valid_results = report.get_current_valid_results()
        test_results = report.get_current_test_results()
        loss = report.epoch_losses[-1]
        updated = False
        if select_model == "last":
            report.current_test_results = test_results
            updated = True
        elif select_model == "best":
            if not datasets.valid_set:
                # there's no valid set, report test result with lowest loss
                if loss <= min(report.epoch_losses):
                    dataset = report.test_sets[0]
                    report.current_test_results = test_results
                    logger.info("Result updated (lowest loss reached: %.4f) - %s" % (
                        loss,
                        ", ".join(["%s: %.2f" % (metric, res) for metric, res in report.current_test_results[dataset].items()])
                    ))
                    model.save_checkpoint("best")
                    updated = True
            else:
                for metric in report.metrics:
                    valid_best_result = max([r[metric] for r in report.epoch_valid_results])
                    if valid_best_result == valid_results[metric]:
                        updated = True
                        if datasets.test_sets:
                            # report test result of best model on valid set
                            # logger.info("Best result: %f", test_result['result'][metric])
                            for name in datasets.test_sets.keys():
                                report.current_test_results[name][metric] = test_results[name][metric]
                                logger.info(f"{name} results updated (better result on valid set: %.4f) - %.4f" % (
                                    valid_results[metric],
                                    test_results[name][metric]
                                ))
                                # log_result(f"valid_test_{metric}", params, test_result, datasets.builder.is_better_result)
                                # log_outputs("valid_test", params, test_outputs)
                        else:
                            pass
                            # there's no test set, report best valid result
                            # log_result(f"valid_{metric}", params, valid_result, datasets.builder.is_better_result)
                            # report.current_results[metric] = valid_results[metric]
                            # log_outputs("valid", params, valid_outputs)
        return updated

    def train(
            self,
            model,
            datasets: Datasets,
            summary_writer: SummaryWriter,
            tqdm_desc="",
            tqdm_position=None,
            on_epoch_finished: Callable[[], None] = None) -> Dict[str, Dict[str, float]]:
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

        epoch = model.global_step // len(datasets.train_set)
        num_samples = model.global_step % len(datasets.train_set)

        training_progress = TrainingProgress(params, num_samples=len(datasets.train_set))

        report.epoch_losses = []
        report.valid_results = dict()
        report.test_results = dict()
        report.current_test_results = {name: {} for name in datasets.test_sets.keys()}
        report.training_progress = training_progress

        # num_samples = 0
        for current_epoch in range(epoch + 1, train_cfg.num_epochs + 1):
            training_progress.new_epoch(current_epoch)
            log_dict = dict(epoch=current_epoch)
            log_dict['total_time'], loss = self.train_epoch(
                current_epoch, model, datasets, report, num_samples,
                training_progress=training_progress,
                tqdm_desc=tqdm_desc + f"Epoch {current_epoch}",
                tqdm_position=tqdm_position)
            report.epoch_losses.append(loss)
            summary_writer.add_scalar(f"loss", loss, current_epoch)
            log_dict['loss'] = loss
            num_samples = 0

            def _evaluate(name, dataset):
                # Evaluate model
                ret = self.evaluate(
                    model, dataset,
                    output_path=os.path.join(params.log_dir, "results"),
                    output_tag="latest",
                    tqdm_desc=tqdm_desc + f"Epoch {current_epoch}",
                    tqdm_position=None if tqdm_position is None else tqdm_position)
                best_result = log_result(name, params, ret.results, datasets.builder.is_better_result)
                return ret.results, best_result, ret.outputs

            if training_progress.should_eval() or current_epoch == train_cfg.num_epochs:
                # Evaluate test sets
                test_results = {}
                for name, dataset in datasets.test_sets.items():
                    test_result, test_best_result, test_outputs = _evaluate(name, dataset)
                    test_results[name] = test_result['result']
                    log_outputs("test", params, test_outputs)
                    log_dict['test_result'] = test_result['result']
                    for metric in test_result['result']:
                        summary_writer.add_scalar(f"{name}_{metric}", test_result['result'][metric], current_epoch)
                report.test_results[current_epoch] = test_results

                # Evaluate valid set
                valid_result = None
                if datasets.valid_set:
                    valid_result, valid_best_result, valid_outputs = _evaluate("valid", datasets.valid_set)
                    log_outputs("valid", params, valid_outputs)
                    log_dict['valid_result'] = valid_result['result']
                    for metric in valid_result['result']:
                        summary_writer.add_scalar(
                            f"valid_{metric}",
                            valid_result['result'][metric], current_epoch)
                    valid_result = valid_result['result']
                report.valid_results[current_epoch] = valid_result

                # results for reporting
                if self.record_results(train_cfg.select_model, model, datasets):
                    report.save()

                if args.output_test_samples:
                    logger.info("Random samples")
                    for output in random.choices(test_outputs if datasets.test_sets else valid_outputs, k=5):
                        logger.info(str(output))

                epoch_info_logger.info(json_dumps(log_dict))
                log_msgs = [
                    "time: %s" % log_dict['total_time'].split('.')[0],
                    "loss: %.4f" % log_dict['loss']
                ]

                for metric in report.metrics:
                    if datasets.valid_set:
                        log_msgs.append(f"dev ({metric}): %.2f" % (
                            log_dict['valid_result'][metric],
                            # valid_best_result[metric]['result'][metric]
                        ))
                    if datasets.test_sets:
                        log_msgs.append(f"test ({metric}): %.2f" % (
                            log_dict['test_result'][metric],
                            # test_best_result[metric]['result'][metric],
                        ))
                logger.info(f"session {report.training_idx} - epoch {current_epoch}: " + " - ".join(log_msgs))

                # Early stopping
                if params.train.early_stop:
                    ne = params.train.early_stop.num_epochs
                    min_diff = params.train.early_stop.min_diff or 0.
                    if datasets.valid_set is not None:
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
                on_epoch_finished()

        return report.current_test_results

    def train_epoch(
            self,
            current_epoch: int,
            model,
            datasets: Datasets,
            report: ModelReport,
            num_samples=0,
            training_progress: TrainingProgress = None,
            tqdm_desc="Epoch {current_epoch}",
            tqdm_position=None):
        """Train."""
        args = self.configs.args
        params = self.params

        if self.params.dataset.shuffle:
            datasets.train_set.shuffle()

        model.reset_counter()
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

        with tqdm(
                desc=tqdm_desc.format(current_epoch=current_epoch),
                total=training_progress.num_samples, leave=False,
                position=tqdm_position,
                disable=not args.show_progress) as t:
            t.update(num_samples)
            batch_size_checkpoints = sorted(batch_sizes.keys())
            for start, end in zip(batch_size_checkpoints, batch_size_checkpoints[1:] + [100]):
                if end / 100 < num_samples / len(datasets.train_set):
                    continue
                batch_size = batch_sizes[start]
                data_train = datasets.train_set.get_iter(
                    batch_size,
                    start=max(start * len(datasets.train_set) // 100, num_samples),
                    end=end * len(datasets.train_set) // 100
                )

                for epoch_step, batch in enumerate(data_train):
                    loss = model.training_step(batch)
                    metrics = model.get_metrics()
                    try:
                        if batch is None or len(batch) == 0:
                            raise Exception("Batch size 0")
                        # loss = model.training_step(batch)
                        # metrics = model.get_metrics()
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
                            **{metric: "%.2f" % val for metric, val in metrics.items()},
                            # **(report.current_results or {})
                        )

                    # if args.debug and epoch_step > DEBUG_NUM_ITERATIONS:
                    #    break
                    t.update(len(batch))
                    training_progress.update(len(batch))

                    model.current_epoch = current_epoch
                    model.global_step = (current_epoch - 1) * len(datasets.train_set) + num_samples

                    if report.summary_writer is not None:
                        report.summary_writer.add_scalar("loss", loss, model.global_step)

                    # Save model
                    if training_progress.should_save():
                        if args.save_all:
                            model.save_checkpoint("epoch-%02d" % current_epoch)
                        else:
                            model.save_checkpoint("latest")

                    # Log
                    if training_progress.should_log():
                        logger.info(", ".join([
                            f"epoch: {current_epoch}",
                            f"progress: {int(prog.epoch_progress * 100)}%",
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
            output_path: str = None,
            output_tag: str = None,
            tqdm_desc="Eval",
            tqdm_position=None) -> EvaluationResults:
        """
        Evaluate model and save result.
        :param model:
        :param dataset:
        :param output_path: path without extension
        :param output_tag:
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
            all_preds, all_refs, sample_ids, extra_all = [], [], [], []
            with tqdm(
                    total=len(dataset),
                    desc=tqdm_desc,
                    leave=False,
                    position=tqdm_position,
                    disable=not self.configs.args.show_progress) as t:
                for batch in data_iter:
                    # noinspection PyBroadException
                    try:
                        if batch is None or len(batch) == 0:
                            raise Exception("Batch size 0")

                        inference_outputs = model.infer(batch)
                        pred, ref, *others = inference_outputs
                        all_preds += pred
                        all_refs += ref
                        if batch.ids:
                            sample_ids += batch.ids

                        t.update(len(batch))
                        # for metric in params.test.metrics:
                        #     if metric == "loss":
                        #         loss = model.get_loss(batch, model_output).item()
                        #         _acc, _total = loss * len(y_pred), len(y_pred)
                        #     else:
                        #         _acc, _total = dataset.evaluate_batch(y_pred, batch, metric=metric)
                        #     acc[metric] += _acc
                        #     total[metric] += _total

                        for i, predicted in enumerate(pred):
                            str_input, str_ground_truth, str_predicted = dataset.format_output(
                                predicted, batch.item(i))
                            outputs.append(dict(
                                input=str_input,
                                reference=str_ground_truth,
                                hypothesis=str_predicted))

                            is_passed, last_log = check_interval_passed(last_log, params.test.log_every)
                            if is_passed:
                                pass
                                # logger.debug(
                                #     "sample %d\n\t[inp] %s\n\t[ref] %s\n\t[hyp] %s",
                                #     len(outputs),
                                #     str(outputs[-1]['input']),
                                #     str(outputs[-1]['reference']),
                                #     str(outputs[-1]['hypothesis']))

                        if report.summary_writer is not None:
                            model.write_summary(report.summary_writer, batch, (pred, others))
                    except Exception:
                        logger.error(traceback.format_exc())

                for metric in params.test.metrics:
                    results[metric] = dataset.evaluate(all_preds, all_refs, metric, output_path)

                if self.params.test.output and output_path:
                    path = dataset.write_results_to_file(
                        all_preds,
                        sample_ids,
                        output_path,
                        output_tag,
                        self.params.test.output)
                    dataset.builder.run_evaluation_script(path)

        result = {
            "epoch": "%.1f" % model.current_epoch,
            "result": {key: results[key] for key in results}
        }

        return EvaluationResults(result, outputs)

    def set_seed(self):
        super().set_seed()
        torch.manual_seed(self.params.random_seed)