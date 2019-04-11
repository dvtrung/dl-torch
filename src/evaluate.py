from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from configs import Configs
from utils.model_utils import get_dataset, get_model, \
    add_result, load_checkpoint
from utils.logging import logger
from utils.utils import init_dirs

def evaluate(model, dataset, params, save_result=False, output=False):
    """Evaluate model and save result."""
    data_loader = DataLoader(
        dataset,
        batch_size=params.test_batch_size or params.batch_size,
        collate_fn=dataset.collate_fn)

    total = 0
    acc = {key: 0. for key in params.metrics}
    for batch in tqdm(data_loader, desc="Eval"):
        y_pred = model.infer(batch)
        for key in params.metrics:
            acc[key] += dataset.evaluate(y_pred, batch, metric=key) * len(y_pred)
        total += len(y_pred)
        if output:
            for predicted, item in zip(y_pred, batch):
                inp = item["word_tags"].cpu().numpy()
                logger.info(dataset.format_output(inp, item, display="word+tag"))
                logger.info(dataset.format_output(predicted, item, display="word+tag"))
                for key in params.metrics:
                    logger.info("%s: %.2f", key, dataset.evaluate(y_pred, batch, metric=key))

    result = {
        "epoch": "%.2f" % model.epoch,
        "result": {key: acc[key] / total for key in acc}
    }
    best_result = add_result(params, result) if save_result else None

    return result, best_result

def main():
    """Main program."""
    configs = Configs(mode="eval")
    configs.parse_args()
    configs.get_params()
    params = configs.params
    args = configs.args

    torch.manual_seed(params.seed)

    dataset_cls = get_dataset(params)
    model_cls = get_model(params)

    # Init dataset
    dataset_cls.prepare()
    dataset_train = dataset_cls("train", params)
    dataset_test = dataset_cls("test", params)

    # Init model
    model = model_cls(params, dataset_train)

    if args.load is None:
        raise Exception("A saved model file must be specified.")
    load_checkpoint(args.load, params, model, None)
    init_dirs(params)

    logger.info("Saved model loaded: %s", args.load)

    res = evaluate(model, dataset_test, params, output=True)

    logger.info(str(res))

if __name__ == "__main__":
    main()
