"""Infer result"""
import torch
from torch.utils.data import DataLoader

from dlex.configs import Configs
from dlex.torch.models.base import DataParellelModel
from dlex.utils.model_utils import get_dataset
from dlex.torch.utils.model_utils import get_model
from dlex.utils.logging import logger
from dlex.utils.utils import init_dirs


def infer(model, dataset, params):
    """Infer results"""

    data_loader = DataLoader(
        dataset,
        batch_size=params.test.batch_size or params.train.batch_size,
        collate_fn=dataset.collate_fn)

    count = 0
    for batch in data_loader:
        y_pred, _, _ = model.infer(batch)
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

    dataset_builder = get_dataset(params)
    dataset = dataset_builder.get_pytorch_wrapper("infer")
    model_cls = get_model(params)

    # Init model
    model = model_cls(params, dataset)
    device_ids = [i for i in range(torch.cuda.device_count())]
    logger.info("Inferring on %s" % str(device_ids))
    model = DataParellelModel(model, device_ids)
    if torch.cuda.is_available():
        logger.info("Cuda available: %s", torch.cuda.get_device_name(0))
        model.cuda()

    model.load_checkpoint(args.load)
    init_dirs(params)

    while True:
        s = input('--> ')
        dataset.load_from_input(s)
        infer(model, dataset, params)


if __name__ == "__main__":
    main()
