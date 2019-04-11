"""Train a model."""

from datetime import datetime
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from configs import Configs
from utils.model_utils import get_model, get_dataset, get_optimizer, \
    load_checkpoint, save_checkpoint
from utils.logging import logger
from utils.utils import init_dirs
from evaluate import evaluate

def train(params, args):
    """Train."""
    torch.manual_seed(params.seed)

    dataset_cls = get_dataset(params)
    model_cls = get_model(params)

    logger.info("Dataset: %s. Model: %s", str(dataset_cls), str(model_cls))

    # Init dataset
    dataset_cls.prepare(force=args.force_preprocessing)
    if args.debug:
        dataset_train = dataset_cls("debug", params)
        dataset_test = dataset_cls("debug", params)
    else:
        dataset_train = dataset_cls("train", params)
        dataset_test = dataset_cls("test", params)

    # Init model
    model = model_cls(params, dataset_train)
    if torch.cuda.is_available():
        logger.info("Cuda available: %s", torch.cuda.get_device_name(0))
        model.cuda()

    # Init optimizer
    optim = get_optimizer(params, model)

    if args.load:
        load_checkpoint(args.load, params, model, optim)
        init_dirs(params)
        logger.info("Saved model loaded: %s", args.load)
        logger.info("Epoch: %f", model.global_step / len(dataset_train))
    else:
        params.set('training_id', datetime.now().strftime('%Y%m%d-%H%M%S'))
        init_dirs(params)

    logger.info("Train size: %d", len(dataset_train))

    data_train = DataLoader(
        dataset_train,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
        collate_fn=dataset_train.collate_fn)

    if not args.train:
        logger.info("Abort without training.")
        return

    logger.info("Training model...")

    epoch = int(model.global_step / len(dataset_train))
    for current_epoch in range(epoch + 1, epoch + params.num_epochs + 1):
        logger.info("--- Epoch %d ---", current_epoch)
        loss_sum = 0

        with tqdm(data_train, desc="Epoch %d" % current_epoch) as t:
            for epoch_step, batch in enumerate(t):
                model.zero_grad()
                loss = model.loss(batch)
                loss.backward()
                optim.step()
                loss_sum += loss.item()
                t.set_postfix(loss=loss.item())

                if args.debug and epoch_step > 10:
                    break

                model.global_step = (current_epoch - 1) * len(dataset_train) + \
                    epoch_step * params.batch_size

        res, best_res = evaluate(model, dataset_test, params, save_result=True)
        for metric in best_res:
            if best_res[metric] == res:
                save_checkpoint(
                    "best" if len(params.metrics) == 1 else "best-%s" % metric,
                    params, model, optim)
                logger.info("Best checkpoint for %s saved", metric)

        logger.info(str(res))
        #logger.info("Loss: %f, Acc: %f" % (loss, res))
        save_checkpoint("epoch-%02d" % current_epoch, params, model, optim)

def main():
    """Read config and train model."""
    configs = Configs(mode="train")
    train(configs.params, configs.args)

if __name__ == "__main__":
    main()
