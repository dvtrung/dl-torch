import os
import torch
# import curses
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from configs import Configs
from utils.model_utils import get_model, get_dataset, get_optimizer, load_checkpoint, save_checkpoint
from utils.logging import logger, set_log_dir
from utils.ops_utils import Tensor, LongTensor

def train(params, args):
    Dataset = get_dataset(params)
    Model = get_model(params)

    Dataset.prepare()

    if args.debug:
        dataset_train = Dataset("debug", params)
        dataset_test = Dataset("debug", params)
    else:
        dataset_train = Dataset("train", params)
        dataset_test = Dataset("test", params)

    model = Model(params, dataset_train)

    if torch.cuda.is_available():
        logger.info("Cuda available: " + torch.cuda.get_device_name(0))
        model.cuda()

    optim = get_optimizer(params, model)

    if args.load:
        load_checkpoint(args.load, params, model, optim)
        logger.info("Saved model loaded: %s" % args.load)
        logger.info("Epoch: %f" % (model.global_step / len(dataset_train)))
        res = eval(model, dataset_test, params)
        logger.info("Evaluate saved model: %f" % res)

    logger.info("Train size: %d" % len(dataset_train))

    data_train = DataLoader(
        dataset_train,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
        collate_fn=dataset_train.collate_fn)

    logger.info("Training model...")

    epoch = int(model.global_step / len(dataset_train))
    for ei in range(epoch + 1, epoch + params.num_epochs + 1):
        logger.info("--- Epoch %d ---" % ei)
        loss_sum = 0

        epoch_step = 0
        for id, batch in enumerate(tqdm(data_train, desc="Epoch %d" % ei)):
            model.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            optim.step()
            loss_sum += loss.item()

            epoch_step += 1
            if args.debug and epoch_step > 10:
                break

            model.global_step = (ei - 1) * len(dataset_train) + id * params.batch_size

        res = eval(model, dataset_test, params)

        logger.info(str(res))
        #logger.info("Loss: %f, Acc: %f" % (loss, res))
        save_checkpoint("epoch-%02d" % ei, params, model, optim)

def eval(model, dataset, params):
    data_loader = DataLoader(
        dataset,
        batch_size=params.test_batch_size or params.batch_size,
        collate_fn=dataset.collate_fn)

    total = 0
    acc = { key: 0. for key in params.metrics }
    for batch in tqdm(data_loader, desc="Eval"):
        y_pred = model.predict(batch)
        for key in params.metrics:
            acc[key] += dataset.eval(y_pred, batch, metric=key)
        total += len(y_pred)

    return { key: acc[key] / total for key in acc }

def main():
    configs = Configs()
    configs.parse_args()
    configs.get_params()
    params = configs.params
    args = configs.args

    torch.manual_seed(params.seed)

    log_dir = os.path.join("logs", args.config_path)
    os.makedirs(log_dir, exist_ok=True)
    set_log_dir(os.path.join(log_dir, datetime.now().strftime('%Y%m%d-%H%M%S')))

    train(params, args)


if __name__ == "__main__":
    main()
