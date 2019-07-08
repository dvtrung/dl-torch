"""Train a model."""
import json
import random
import time
from collections.abc import Iterator
from datetime import datetime

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch
from dlex.configs import Configs, AttrDict
from dlex.datasets.torch import PytorchDataset
from dlex.torch.evaluate import evaluate
from dlex.torch.models.base import DataParellelModel, BaseModel
from dlex.torch.utils.model_utils import get_model, \
    load_checkpoint, save_checkpoint
from dlex.utils.logging import logger, epoch_info_logger, epoch_step_info_logger, logging, log_result
from dlex.utils.model_utils import get_dataset
from dlex.utils.utils import init_dirs

DEBUG_NUM_ITERATIONS = 5
DEBUG_BATCH_SIZE = 4


def train(
        params: AttrDict,
        args,
        model: BaseModel,
        dataset_train: PytorchDataset,
        dataset_test: PytorchDataset,
        summary_writer):
    epoch = model.global_step // len(dataset_train)
    num_samples = model.global_step % len(dataset_train)
    # num_samples = 0
    for current_epoch in range(epoch + 1, epoch + params.train.num_epochs + 1):
        total_time, result, best_result, loss = train_epoch(
            current_epoch, params, args,
            model, dataset_train, dataset_test, summary_writer, num_samples)
        num_samples = 0
        epoch_info_logger.info(json.dumps(dict(
            epoch=current_epoch,
            total_time=str(total_time),
            result=result['result'],
            best_result=best_result,
            loss=loss
        )))


def get_data_iterator(
        dataset: PytorchDataset,
        batch_size: int,
        start=0, end=-1) -> Iterator:
    return DataLoader(
        dataset[start:end],
        batch_size=batch_size,
        collate_fn=dataset.collate_fn)


def check_interval_passed(last_done: float, interval: str, progress) -> (bool, float):
    unit = interval[-1]
    value = float(interval[:-1])
    if unit == "e":  # epoch progress (percentage)
        if progress - last_done >= value:
            return True, progress
        else:
            return False, last_done
    elif unit == "s":
        if time.time() - last_done > value:
            return True, time.time()
        else:
            return False, last_done


def train_epoch(
        current_epoch: int,
        params: AttrDict,
        args,
        model: BaseModel,
        dataset_train: PytorchDataset,
        dataset_test: PytorchDataset,
        summary_writer,
        num_samples=0):
    """Train."""
    if params.dataset.shuffle:
        dataset_train.shuffle()

    logger.info("EPOCH %d", current_epoch)
    loss_sum, loss_count = 0, 0
    start_time = datetime.now()

    if isinstance(params.train.batch_size, int):  # fixed batch size
        batch_sizes = {0: params.train.batch_size}
    else:
        batch_sizes = params.train.batch_size

    assert 0 in batch_sizes

    total = len(dataset_train)
    last_save = 0
    last_log = 0
    with tqdm(desc="Epoch %d" % current_epoch, total=total) as t:
        t.update(num_samples)
        batch_size_checkpoints = sorted(batch_sizes.keys())
        for start, end in zip(batch_size_checkpoints, batch_size_checkpoints[1:] + [100]):
            if end / 100 < num_samples / len(dataset_train):
                continue
            batch_size = batch_sizes[start]
            logger.info("Batch size: %d", batch_size)
            data_train = get_data_iterator(
                dataset_train, batch_size,
                start=max(start * len(dataset_train) // 100, num_samples),
                end=end * len(dataset_train) // 100
            )

            for epoch_step, batch in enumerate(data_train):
                try:
                    loss = model.training_step(batch)
                    loss_sum += loss.item()
                    loss_count += 1
                except RuntimeError as e:
                    logger.error(str(e))
                t.set_postfix(loss=loss.item())

                # if args.debug and epoch_step > DEBUG_NUM_ITERATIONS:
                #    break
                t.update(batch_size)
                num_samples += batch_size
                progress = 1. if total - num_samples < batch_size else num_samples / total

                model.current_epoch = current_epoch
                model.global_step = (current_epoch - 1) * len(dataset_train) + num_samples

                if summary_writer is not None:
                    summary_writer.add_scalar("loss", loss, model.global_step)

                # Save model
                is_passed, last_save = check_interval_passed(last_save, params.train.save_every, progress)
                if is_passed:
                    logger.info("Saving checkpoint...")
                    if args.save_all:
                        save_checkpoint("epoch-%02d" % current_epoch, params, model)
                    else:
                        save_checkpoint("latest", params, model)

                # Log
                is_passed, last_log = check_interval_passed(last_log, params.train.log_every, progress)
                if is_passed:
                    epoch_step_info_logger.info(json.dumps(dict(
                        epoch=current_epoch + progress - 1,
                        loss=loss.item(),
                        overall_loss=loss_sum / loss_count
                    )))
    save_checkpoint("epoch-latest", params, model)
    end_time = datetime.now()

    # Evaluate model
    result, outputs = evaluate(
        model,
        dataset_test,
        params,
        output=True,
        summary_writer=summary_writer)
    best_result = log_result(params, result, dataset_train.builder.is_better_result)

    for metric in result['result']:
        if summary_writer is not None:
            summary_writer.add_scalar("eval_%s" % metric, result['result'][metric], current_epoch)

    logger.info("Random samples")
    for output in random.choices(outputs, k=5):
        logger.info(str(output))
        logger.info('')

    for metric in best_result:
        if best_result[metric] == result:
            save_checkpoint(
                "best" if len(params.test.metrics) == 1 else "best-%s" % metric,
                params, model)
            logger.info("Best checkpoint for %s saved", metric)

    return end_time - start_time, result, best_result, loss_sum / loss_count


def main(argv=None):
    """Read config and train model."""
    configs = Configs(mode="train", argv=argv)
    params, args = configs.params, configs.args

    if args.debug:
        params.train.batch_size = DEBUG_BATCH_SIZE
        params.test.batch_size = DEBUG_BATCH_SIZE

    torch.manual_seed(params.seed)

    # Init dataset
    dataset_builder = get_dataset(params)
    assert dataset_builder
    if not args.no_prepare:
        dataset_builder.prepare(download=args.download, preprocess=args.preprocess)
    if args.debug:
        dataset_train = dataset_builder.get_pytorch_wrapper("debug")
        dataset_test = dataset_builder.get_pytorch_wrapper("debug")
    else:
        dataset_train = dataset_builder.get_pytorch_wrapper("train")
        dataset_test = dataset_builder.get_pytorch_wrapper("test")

    # Init model
    model_cls = get_model(params)
    assert model_cls
    model = model_cls(params, dataset_train)

    for parameter in model.parameters():
        logger.debug(parameter.shape)

    device_ids = [i for i in range(torch.cuda.device_count())]
    logger.info("Training on %s" % str(device_ids))
    model = DataParellelModel(model, device_ids)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load checkpoint or initialize new training
    if args.load:
        load_checkpoint(args.load, params, model)
        init_dirs(params)
        logger.info("Saved model loaded: %s", args.load)
        logger.info("Epoch: %f", model.global_step / len(dataset_train))
    else:
        params.set('training_id', datetime.now().strftime('%Y%m%d-%H%M%S'))
        init_dirs(params)

    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.info("Dataset: %s. Model: %s", str(dataset_builder), str(model_cls))
    if use_cuda:
        logger.info("CUDA available: %s", torch.cuda.get_device_name(0))

    logger.info("Training started.")

    # summary_writer = SummaryWriter()
    summary_writer = None

    if args.num_processes == 1:
        train(configs.params, configs.args, model, dataset_train, dataset_test, summary_writer=summary_writer)
    else:
        model.share_memory()
        # TODO: Implement multiprocessing
        mp.set_start_method('spawn')
        processes = []
        for rank in range(args.num_processes):
            p = mp.Process(target=train, args=(model, dataset_train, dataset_test))
            # We first train the model across `num_processes` processes
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
