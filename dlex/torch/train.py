"""Train a model."""

from datetime import datetime
import random

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dlex.configs import Configs
from dlex.torch.evaluate import evaluate
from dlex.utils.logging import logger, logging
from dlex.torch.utils.model_utils import get_model, get_dataset, \
    load_checkpoint, save_checkpoint
from dlex.utils.utils import init_dirs

DEBUG_NUM_ITERATIONS = 5
DEBUG_BATCH_SIZE = 4


def train(params, args, model, dataset_train, dataset_test, summary_writer):
    data_train = DataLoader(
        dataset_train,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
        collate_fn=dataset_train.collate_fn,
        pin_memory=params.cpu)

    epoch = int(model.global_step / len(dataset_train))
    for current_epoch in range(epoch + 1, epoch + params.num_epochs + 1):
        train_epoch(current_epoch, params, args, model, data_train, dataset_test, summary_writer)


def train_epoch(current_epoch, params, args, model, data_train, dataset_test, summary_writer):
    """Train."""
    logger.info("EPOCH %d", current_epoch)
    loss_sum, loss_count = 0, 0
    start_time = datetime.now()
    with tqdm(data_train, desc="Epoch %d" % current_epoch) as t:
        for epoch_step, batch in enumerate(t):
            loss = model.training_step(batch)
            loss_sum += loss.item()
            loss_count += 1
            t.set_postfix(loss=loss_sum / loss_count)

            #if args.debug and epoch_step > DEBUG_NUM_ITERATIONS:
            #    break

            model.current_epoch = current_epoch
            model.global_step = (current_epoch - 1) * len(data_train.dataset) + \
                epoch_step * params.batch_size
            summary_writer.add_scalar("loss", loss, model.global_step)
    end_time = datetime.now()
    if args.save_all:
        save_checkpoint("epoch-%02d" % current_epoch, params, model)
    res, best_res, outputs = evaluate(model, dataset_test, params, save_result=True, output=True, summary_writer=summary_writer)

    for metric in res['result']:
        summary_writer.add_scalar("eval_%s" % metric, res['result'][metric], current_epoch)

    logger.info("Random samples")
    for output in random.choices(outputs, k=5):
        logger.info(output)
        logger.info('')

    for metric in best_res:
        if best_res[metric] == res:
            save_checkpoint(
                "best" if len(params.metrics) == 1 else "best-%s" % metric,
                params, model)
            logger.info("Best checkpoint for %s saved", metric)

    logger.info("Epoch time: %s", str(end_time - start_time))
    logger.info("Eval: %s", str(res['result']))
    logger.info("-------")


def main(argv=None):
    """Read config and train model."""
    configs = Configs(mode="train", argv=argv)
    params, args = configs.params, configs.args

    if args.debug:
        params.batch_size = DEBUG_BATCH_SIZE
        params.eval_batch_size = DEBUG_BATCH_SIZE

    torch.manual_seed(params.seed)

    # Init dataset
    dataset_builder = get_dataset(params)
    assert dataset_builder
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

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

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
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.num_processes == 1:
        train(configs.params, configs.args, model, dataset_train, dataset_test, summary_writer=SummaryWriter())
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