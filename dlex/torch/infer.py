"""Infer result"""
import torch
from torch.utils.data import DataLoader

from configs import Configs
from torch.utils.model_utils import get_dataset, get_model, load_checkpoint
from utils.logging import logger
from utils.utils import init_dirs


def infer(model, dataset, params):
    """Infer results"""

    data_loader = DataLoader(
        dataset,
        batch_size=params.test.batch_size or params.train.batch_size,
        collate_fn=dataset.collate_fn)

    count = 0
    for batch in data_loader:
        y_pred = model.infer(batch)
        for i, _y in enumerate(y_pred):
            count += 1
            logger.info(dataset.format_output(
                _y,
                batch.item(i)))
        # logger.info('\n'.join([str(r) for r in ret]))


def main():
    """Main program"""
    configs = Configs(mode="infer")
    params = configs.params
    args = configs.args
    torch.manual_seed(params.seed)

    dataset_cls = get_dataset(params)
    model_cls = get_model(params)

    # Init dataset
    dataset_infer = dataset_cls("infer", params)

    # Init model
    model = model_cls(params, dataset_infer)
    if torch.cuda.is_available():
        logger.info("Cuda available: %s", torch.cuda.get_device_name(0))
        model.cuda()

    load_checkpoint(args.load, params, model)
    init_dirs(params)

    if args.input:
        dataset_infer.load_from_input(args.input)
        infer(model, dataset_infer, params)
    else:
        s = input('--> ')
        dataset_infer.load_from_input(s)
        infer(model, dataset_infer, params)


if __name__ == "__main__":
    main()
