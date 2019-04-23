"""Train a model."""

from datetime import datetime
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from .configs import Configs
from .utils.model_utils import get_model, get_dataset, \
    load_checkpoint, save_checkpoint
from .utils.logging import logger
from .utils.utils import init_dirs
from .evaluate import evaluate


def train(params, args, model, dataset_train, dataset_test):
    data_train = DataLoader(
        dataset_train,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
        collate_fn=dataset_train.collate_fn,
        pin_memory=params.cuda)

    epoch = int(model.global_step / len(dataset_train))
    for current_epoch in range(epoch + 1, epoch + params.num_epochs + 1):
        train_epoch(current_epoch, params, args, model, data_train, dataset_test)


def train_epoch(current_epoch, params, args, model, data_train, dataset_test):
    """Train."""
    logger.info("--- Epoch %d ---", current_epoch)
    loss_sum = 0

    with tqdm(data_train, desc="Epoch %d" % current_epoch) as t:
        for epoch_step, batch in enumerate(t):
            loss = model.training_step(batch)
            loss_sum += loss.item()
            t.set_postfix(loss=loss.item())

            if args.debug and epoch_step > 10:
                break

            model.global_step = (current_epoch - 1) * len(data_train.dataset) + \
                epoch_step * params.batch_size

    res, best_res = evaluate(model, dataset_test, params, save_result=True)
    for metric in best_res:
        if best_res[metric] == res:
            save_checkpoint(
                "best" if len(params.metrics) == 1 else "best-%s" % metric,
                params, model)
            logger.info("Best checkpoint for %s saved", metric)

    logger.info(str(res))
    # logger.info("Loss: %f, Acc: %f" % (loss, res))
    save_checkpoint("epoch-%02d" % current_epoch, params, model)


def main(argv=None):
    """Read config and train model."""
    configs = Configs(mode="train", argv=argv)
    params, args = configs.params, configs.args

    torch.manual_seed(params.seed)

    # Init dataset
    dataset_cls = get_dataset(params)
    assert dataset_cls
    dataset_cls.prepare(download=args.download, preprocess=args.preprocess)
    if args.debug:
        dataset_train = dataset_cls("debug", params)
        dataset_test = dataset_cls("debug", params)
    else:
        dataset_train = dataset_cls("train", params)
        dataset_test = dataset_cls("test", params)

    # Init model
    model_cls = get_model(params)
    assert model_cls
    model = model_cls(params, dataset_train)

    logger.info("Dataset: %s. Model: %s", str(dataset_cls), str(model_cls))

    if torch.cuda.is_available():
        logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
        model.cuda()

    if args.load:
        load_checkpoint(args.load, params, model)
        init_dirs(params)
        logger.info("Saved model loaded: %s", args.load)
        logger.info("Epoch: %f", model.global_step / len(dataset_train))
    else:
        params.set('training_id', datetime.now().strftime('%Y%m%d-%H%M%S'))
        init_dirs(params)

    logger.info("Train size: %d", len(dataset_train))

    logger.info("Training model...")
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.num_processes == 1:
        train(configs.params, configs.args, model, dataset_train, dataset_test)
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
