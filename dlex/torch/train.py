"""Train a model."""
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
from dlex.configs import MainConfig, Configs
from dlex.datatypes import ModelReport
from dlex.torch.datatypes import Datasets
from dlex.torch.evaluate import evaluate
from dlex.torch.models.base import DataParellelModel
from dlex.torch.utils.utils import load_model, set_seed
from dlex.utils.logging import logger, epoch_info_logger, epoch_step_info_logger, log_result, json_dumps, \
    log_outputs
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

DEBUG_NUM_ITERATIONS = 5


def train(
        params: MainConfig,
        configs: Configs,
        model: DataParellelModel,
        datasets: Datasets,
        report: ModelReport,
        summary_writer: SummaryWriter,
        tqdm_desc="",
        tqdm_position=None):
    args = configs.args

    epoch = model.global_step // len(datasets.train)
    num_samples = model.global_step % len(datasets.train)
    report.results = {}

    report.num_epochs = params.train.num_epochs
    report.epoch_losses = []
    report.epoch_valid_results = []
    report.epoch_test_results = []

    # num_samples = 0
    for current_epoch in range(epoch + 1, params.train.num_epochs + 1):
        log_dict = dict(epoch=current_epoch)
        log_dict['total_time'], loss = train_epoch(
            current_epoch, params, configs,
            model, datasets, report, num_samples,
            tqdm_desc=tqdm_desc + f"Epoch {current_epoch}",
            tqdm_position=tqdm_position)
        report.epoch_losses.append(loss)
        summary_writer.add_scalar(f"loss", loss, current_epoch)
        log_dict['loss'] = loss
        num_samples = 0

        def _evaluate(mode):
            # Evaluate model
            result, outputs = evaluate(
                model, getattr(datasets, mode), params, configs,
                output_path=os.path.join(params.log_dir, "results", "latest"),
                report=report,
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
            if datasets.valid is None:
                # there's no valid set, report test result with lowest loss
                if loss <= min(report.epoch_losses):
                    report.results = test_result['result']
                    logger.info("Result updated (lowest loss reached: %.4f) - %s" % (
                        loss,
                        ", ".join(["%s: %.2f" % (metric, res) for metric, res in report.results.items()])
                    ))

        if datasets.valid is not None:
            valid_result, valid_best_result, valid_outputs = _evaluate("valid")
            report.epoch_valid_results.append(valid_result['result'])
            log_outputs("valid", params, valid_outputs)
            log_dict['valid_result'] = valid_result['result']
            for metric in valid_result['result']:
                summary_writer.add_scalar(
                    f"valid_{metric}",
                    valid_result['result'][metric], current_epoch)

            for metric in valid_best_result:
                if valid_best_result[metric] == valid_result:
                    if datasets.test is not None:
                        # report test result of best model on valid set
                        logger.info("Best result: %f", test_result['result'][metric])
                        report.results[metric] = test_result['result'][metric]
                        log_result(f"valid_test_{metric}", params, test_result, datasets.train.builder.is_better_result)
                        log_outputs("valid_test", params, test_outputs)
                    else:
                        # there's no test set, report best valid result
                        log_result(f"valid_{metric}", params, valid_result, datasets.train.builder.is_better_result)
                        report.results[metric] = valid_result['result'][metric]
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

        for metric in params.test.metrics:
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
        logger.info(f"[Epoch {current_epoch}] " + " - ".join(log_msgs))

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
                            for metric in params.test.metrics):
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

    return report.results


def check_interval_passed(last_done: float, interval: str, progress) -> (bool, float):
    unit = interval[-1]
    value = float(interval[:-1])
    if unit == "e":  # epoch progress (percentage)
        if progress - last_done >= value:
            return True, progress
        else:
            return False, last_done
    elif unit in ["s", "m", "h"]:
        d = dict(s=1, m=60, h=3600)[unit]
        if time.time() / d - last_done > value:
            return True, time.time() / d
        else:
            return False, last_done


def train_epoch(
        current_epoch: int,
        params: MainConfig,
        configs: Configs,
        model: DataParellelModel,
        datasets: Datasets,
        report: ModelReport,
        num_samples=0,
        tqdm_desc="Epoch {current_epoch}",
        tqdm_position=None):
    """Train."""
    report.current_epoch = current_epoch
    args = configs.args

    if params.dataset.shuffle:
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
        batch_sizes[key] *= (len(params.gpu) if params.gpu else 1) or 1
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
                        loss=loss,
                        epoch_loss=model.epoch_loss,
                        # lr=mean(model.learning_rates())
                        **(report.results or {})
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
                    epoch_step_info_logger.info(json_dumps(dict(
                        epoch=current_epoch + progress - 1,
                        loss=loss,
                        overall_loss=model.epoch_loss
                    )))

                if args.debug:
                    input("Press any key to continue...")
            model.end_training_epoch()
    # model.save_checkpoint("epoch-latest")
    end_time = datetime.now()
    return str(end_time - start_time), model.epoch_loss


def main(
        argv=None,
        params=None,
        configs: Configs = None,
        training_idx: int = None,
        report_queue = None):
    """Read config and train model."""
    logger.info(f"Training started ({training_idx}).")
    report = ModelReport()
    report.metrics = params.test.metrics
    report.results = {m: None for m in params.test.metrics}

    if params.train.cross_validation:
        set_seed(params.random_seed)
        results = []
        for i in range(params.train.cross_validation):
            summary_writer = SummaryWriter(os.path.join(configs.log_dir, "runs", str(training_idx), str(i + 1)))
            report.cv_current_fold = i + 1
            report.cv_num_folds = params.train.cross_validation
            params.dataset.cv_current_fold = i + 1
            params.dataset.cross_validation = params.train.cross_validation
            report_queue.put((training_idx, report))

            params, args, model, datasets = load_model("train", report, argv, params, configs)

            res = train(
                params, configs, model, datasets,
                report=report, summary_writer=summary_writer,
                tqdm_desc=f"[{training_idx}] CV {report.cv_current_fold}/{report.cv_num_folds} - ",
                tqdm_position=training_idx)
            results.append(res)
            report.results = {
                metric: [r[metric] for r in results] for metric in results[0]}
            report_queue.put((training_idx, report))

        report.finished = True
        summary_writer.close()
    else:
        summary_writer = SummaryWriter(os.path.join(configs.log_dir, "runs", str(training_idx)))
        set_seed(params.random_seed)
        params, args, model, datasets = load_model("train", report, argv, params, configs)
        res = train(
            params, configs, model, datasets,
            report, summary_writer,
            tqdm_position=training_idx * 3 + 1)
        report.results = res
        report.finished = True

    return report

    #else:
    #    model.share_memory()
    #    # TODO: Implement multiprocessing
    #    mp.set_start_method('spawn')
    #    processes = []
    #    for rank in range(args.num_processes):
    #        p = mp.Process(target=train, args=(model, datasets))
    #        # We first train the model across `num_processes` processes
    #        p.start()
    #        processes.append(p)
    #    for p in processes:
    #        p.join()


if __name__ == "__main__":
    main()
