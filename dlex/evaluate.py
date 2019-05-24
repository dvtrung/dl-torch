from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from .configs import Configs
from .utils.model_utils import get_dataset, get_model, \
    add_result, load_checkpoint
from .utils.logging import logger
from .utils.utils import init_dirs


def evaluate(model, dataset, params, save_result=False, output=False):
    """Evaluate model and save result."""
    data_loader = DataLoader(
        dataset,
        batch_size=params.test_batch_size or params.batch_size,
        collate_fn=dataset.collate_fn)

    total = {key: 0 for key in params.metrics}
    acc = {key: 0. for key in params.metrics}
    outputs = []
    for batch in tqdm(data_loader, desc="Eval"):
        y_pred, y_pred_len = model.infer(batch)
        for key in params.metrics:
            _acc, _total = dataset.evaluate(y_pred, batch, metric=key)
            acc[key] += _acc
            total[key] += _total
        if output:
            for i, predicted in enumerate(y_pred):
                str_input, str_ground_truth, str_predicted = dataset.format_output(
                    predicted, dataset.get_item_from_batch(batch, i))
                outputs.append('\n'.join([str_input, str_ground_truth, str_predicted]))

    result = {
        "epoch": "%.1f" % model.epoch,
        "result": {key: acc[key] / total[key] for key in acc}
    }
    best_result = add_result(params, result) if save_result else None

    return result, best_result, outputs


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
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
        model.cuda()

    if args.load is None:
        raise Exception("A saved model file must be specified.")
    load_checkpoint(args.load, params, model)
    init_dirs(params)

    logger.info("Saved model loaded: %s", args.load)

    result, best_result, outputs = evaluate(model, dataset_test, params, output=True)

    #for output in outputs:
        #logger.info(output)
        #logger.info("---")

    logger.info(str(result))


if __name__ == "__main__":
    main()
