"""Train a model."""
import random
import sys
import time
from datetime import datetime

import torch
import torch.multiprocessing as mp
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dlex.configs import AttrDict
from dlex.torch.datatypes import Datasets
from dlex.torch.evaluate import evaluate
from dlex.torch.models.base import DataParellelModel
from dlex.torch.utils.utils import load_model
from dlex.utils.logging import logger, epoch_info_logger, epoch_step_info_logger, logging, log_result, json_dumps, \
    log_outputs

DEBUG_NUM_ITERATIONS = 5


def train(
        params: AttrDict,
        args,
        model: DataParellelModel,
        datasets: Datasets,
        summary_writer):
    epoch = model.global_step // len(datasets.train)
    num_samples = model.global_step % len(datasets.train)
    # num_samples = 0
    for current_epoch in range(epoch + 1, epoch + params.train.num_epochs + 1):
        log_dict = dict(epoch=current_epoch)
        log_dict['total_time'], log_dict['loss'] = train_epoch(
            current_epoch, params, args,
            model, datasets, summary_writer, num_samples)
        num_samples = 0

        def _evaluate(mode):
            # Evaluate model
            result, outputs = evaluate(
                model,
                getattr(datasets, mode),
                params,
                output=True,
                summary_writer=summary_writer)
            best_result = log_result(mode, params, result, datasets.train.builder.is_better_result)
            for metric in best_result:
                if best_result[metric] == result:
                    model.save_checkpoint(
                        "best" if len(params.test.metrics) == 1 else "%s-best-%s" % (mode, metric))
                    logger.info("Best %s for %s set reached: %f", metric, mode, result['result'][metric])
            return result, best_result, outputs

        test_result, _, test_outputs = _evaluate("test")
        log_outputs("test", params, test_outputs)
        log_dict['test_result'] = test_result['result']
        if datasets.valid is not None:
            valid_result, valid_best_result, valid_outputs = _evaluate("valid")
            log_outputs("valid", params, valid_outputs)
            log_dict['valid_result'] = valid_result['result']
            for metric in valid_best_result:
                if valid_best_result[metric] == valid_result:
                    logger.info("Best result: %f", test_result['result'][metric])
                    log_result(f"valid_test_{metric}", params, test_result, datasets.train.builder.is_better_result)
                    log_outputs("valid_test", params, test_outputs)

        for metric in test_result['result']:
            if summary_writer is not None:
                summary_writer.add_scalar("eval_%s" % metric, test_result['result'][metric], current_epoch)

        logger.info("Random samples")
        for output in random.choices(test_outputs, k=5):
            logger.info(str(output))

        epoch_info_logger.info(json_dumps(log_dict))
        logger.info(json_dumps(log_dict))


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
        params: AttrDict,
        args,
        model: DataParellelModel,
        datasets: Datasets,
        summary_writer,
        num_samples=0):
    """Train."""
    if params.dataset.shuffle:
        datasets.train.shuffle()

    logger.info("EPOCH %d", current_epoch)
    model.start_calculating_loss()
    start_time = datetime.now()

    if isinstance(params.train.batch_size, int):  # fixed batch size
        batch_sizes = {0: params.train.batch_size}
    else:
        batch_sizes = params.train.batch_size
    for key in batch_sizes:
        batch_sizes[key] *= max(torch.cuda.device_count(), 1)
    assert 0 in batch_sizes

    total = len(datasets.train)
    last_save = 0
    last_log = 0
    with tqdm(desc="Epoch %d" % current_epoch, total=total) as t:
        t.update(num_samples)
        batch_size_checkpoints = sorted(batch_sizes.keys())
        for start, end in zip(batch_size_checkpoints, batch_size_checkpoints[1:] + [100]):
            if end / 100 < num_samples / len(datasets.train):
                continue
            batch_size = batch_sizes[start]
            logger.info("Batch size: %d", batch_size)
            data_train = datasets.train.get_iter(
                batch_size,
                start=max(start * len(datasets.train) // 100, num_samples),
                end=end * len(datasets.train) // 100
            )

            for epoch_step, batch in enumerate(data_train):
                try:
                    if batch is None or batch.X.shape[0] == 0:
                        raise Exception("Batch size 0")
                    loss = model.training_step(batch)
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

                if summary_writer is not None:
                    summary_writer.add_scalar("loss", loss, model.global_step)

                # Save model
                is_passed, last_save = check_interval_passed(last_save, params.train.save_every, progress)
                if is_passed:
                    logger.info("Saving checkpoint...")
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


def main(argv=None):
    mp.set_start_method('spawn', force=True)
    """Read config and train model."""
    params, args, model, datasets = load_model("train", argv)
    torch.manual_seed(params.seed)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.info("Training started.")
    # summary_writer = SummaryWriter()
    summary_writer = None

    if args.num_processes == 1:
        train(params, args, model, datasets, summary_writer=summary_writer)
    else:
        model.share_memory()
        # TODO: Implement multiprocessing
        mp.set_start_method('spawn')
        processes = []
        for rank in range(args.num_processes):
            p = mp.Process(target=train, args=(model, datasets))
            # We first train the model across `num_processes` processes
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
