"""Train a model."""
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, Callable

import torch
import torch.multiprocessing as mp
from dlex.configs import MainConfig
from dlex.datatypes import ModelReport
from dlex.torch.datatypes import Datasets
from dlex.torch.evaluate import evaluate
from dlex.torch.models.base import DataParellelModel
from dlex.torch.utils.utils import load_model, set_seed
from dlex.utils.logging import logger, epoch_info_logger, epoch_step_info_logger, log_result, json_dumps, \
    log_outputs
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

DEBUG_NUM_ITERATIONS = 5


def train(
        params: MainConfig,
        args,
        model: DataParellelModel,
        datasets: Datasets,
        report: ModelReport,
        report_callback: Callable[[Dict[str, float], bool], None] = None):
    epoch = model.global_step // len(datasets.train)
    num_samples = model.global_step % len(datasets.train)
    display_results = {}

    # num_samples = 0
    for current_epoch in range(epoch + 1, epoch + params.train.num_epochs + 1):
        log_dict = dict(epoch=current_epoch)
        log_dict['total_time'], log_dict['loss'] = train_epoch(
            current_epoch, params, args,
            model, datasets, report, num_samples)
        num_samples = 0

        def _evaluate(mode):
            # Evaluate model
            result, outputs = evaluate(
                model, getattr(datasets, mode), params,
                output_path=os.path.join(params.log_dir, "results", "latest"),
                report=report)
            best_result = log_result(mode, params, result, datasets.train.builder.is_better_result)
            for metric in best_result:
                if best_result[metric] == result:
                    model.save_checkpoint(
                        "best" if len(params.test.metrics) == 1 else "%s-best-%s" % (mode, metric))
                    logger.info("Best %s for %s set reached: %f", metric, mode, result['result'][metric])
            return result, best_result, outputs

        if datasets.test is not None:
            test_result, test_best_result, test_outputs = _evaluate("test")
            log_outputs("test", params, test_outputs)
            log_dict['test_result'] = test_result['result']
            if datasets.valid is None:
                for metric in test_best_result:
                    display_results[metric] = test_best_result[metric]['result'][metric]

        if datasets.valid is not None:
            valid_result, valid_best_result, valid_outputs = _evaluate("valid")
            log_outputs("valid", params, valid_outputs)
            log_dict['valid_result'] = valid_result['result']
            for metric in valid_best_result:
                if valid_best_result[metric] == valid_result:
                    if datasets.test is not None:
                        logger.info("Best result: %f", test_result['result'][metric])
                        display_results[metric] = test_result['result'][metric]
                        log_result(f"valid_test_{metric}", params, test_result, datasets.train.builder.is_better_result)
                        log_outputs("valid_test", params, test_outputs)
                    else:
                        log_result(f"valid_{metric}", params, valid_result, datasets.train.builder.is_better_result)
                        log_outputs("valid", params, valid_outputs)

        if report.summary_writer:
            for metric in report.metrics:
                report.summary_writer.add_scalar("eval_%s" % metric, test_result['result'][metric], current_epoch)

        if args.output_test_samples:
            logger.info("Random samples")
            for output in random.choices(test_outputs if datasets.test is not None else valid_outputs, k=5):
                logger.info(str(output))

        epoch_info_logger.info(json_dumps(log_dict))
        logger.info("Epoch %d - Total Time: %s - Loss: %.4f" % (
            current_epoch,
            log_dict['total_time'],
            log_dict['loss']
        ))

        for metric in params.test.metrics:
            if datasets.valid:
                logger.info("Dev Result (%s) - %.4f (best: %.4f)" % (
                    metric,
                    log_dict['valid_result'][metric],
                    valid_best_result[metric]['result'][metric]
                ))
            if datasets.test:
                logger.info("Test Result (%s) - %.4f (best: %.4f)" % (
                    metric,
                    log_dict['test_result'][metric],
                    test_best_result[metric]['result'][metric],
                ))

        if report_callback:  # update partial results
            report.results = display_results
            report_callback(report)

    return display_results


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
        args,
        model: DataParellelModel,
        datasets: Datasets,
        report: ModelReport,
        num_samples=0):
    """Train."""

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
    with tqdm(desc="Epoch %d" % current_epoch, total=total, leave=False) as t:
        t.update(num_samples)
        batch_size_checkpoints = sorted(batch_sizes.keys())
        for start, end in zip(batch_size_checkpoints, batch_size_checkpoints[1:] + [100]):
            if end / 100 < num_samples / len(datasets.train):
                continue
            batch_size = batch_sizes[start]
            logger.debug("Batch size: %d", batch_size)
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
                    t.set_postfix(loss=loss, epoch_loss=model.epoch_loss)

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
    model.save_checkpoint("epoch-latest")
    end_time = datetime.now()
    return str(end_time - start_time), model.epoch_loss


def main(
        argv=None,
        params=None,
        args=None,
        report_callback: Callable[[Dict[str, float], bool], None] = None):
    mp.set_start_method('spawn', force=True)
    """Read config and train model."""

    logger.debug("Training started.")
    # summary_writer = SummaryWriter()
    report = ModelReport()
    if args.report:
        report_callback(report)

    if args.num_processes == 1:
        if params.train.cross_validation:
            set_seed(params.random_seed)
            results = []
            for i in tqdm(range(params.train.cross_validation), desc="Cross Validation"):
                report.cross_validation_fold = i
                report.cross_validation_total = params.train.cross_validation
                params.dataset.cross_validation_fold = i
                params.dataset.cross_validation = params.train.cross_validation
                params, args, model, datasets = load_model("train", report, argv, params, args)

                res = train(params, args, model, datasets, report=report, report_callback=None)
                results.append(res)
                if report_callback:
                    report.results = {
                        metric: sum([r[metric] for r in results]) / len(results) for metric in results[0]}
                    report_callback(report)
            if report_callback:
                report.finished = True
                report_callback(report)
        else:
            set_seed(params.random_seed)
            params, args, model, datasets = load_model("train", report, argv, params, args)
            res = train(params, args, model, datasets, report, report_callback=report_callback)
            if report_callback:
                report.results = res
                report.finished = True
                report_callback(report)
        return res
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
