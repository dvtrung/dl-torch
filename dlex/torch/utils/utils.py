from datetime import datetime

import torch

from dlex.configs import Configs
from dlex.torch.datatypes import Datasets
from dlex.torch.models.base import DataParellelModel
from dlex.torch.utils.model_utils import get_model
from dlex.utils import logger, init_dirs
from dlex.utils.model_utils import get_dataset

DEBUG_BATCH_SIZE = 4


def load_model(mode, argv=None, params=None, args=None):
    if not params and not args:
        configs = Configs(mode=mode, argv=argv)
        params_list, args = configs.params_list, configs.args

        variables, params = params_list[0]

    if mode == "train":
        if args.debug:
            params.train.batch_size = DEBUG_BATCH_SIZE
            params.test.batch_size = DEBUG_BATCH_SIZE

    # Init dataset
    dataset_builder = get_dataset(params)
    assert dataset_builder
    if not args.no_prepare:
        dataset_builder.prepare(download=args.download, preprocess=args.preprocess)
    if mode == "test":
        datasets = Datasets(test=dataset_builder.get_pytorch_wrapper(args.eval_set))
    elif mode == "train":
        if args.debug:
            datasets = Datasets(
                train=dataset_builder.get_pytorch_wrapper("test"),
                test=dataset_builder.get_pytorch_wrapper("test"))
        else:
            datasets = Datasets(
                train=dataset_builder.get_pytorch_wrapper("train"),
                valid=dataset_builder.get_pytorch_wrapper("valid") if "valid" in params.train.eval else
                dataset_builder.get_pytorch_wrapper("dev") if "dev" in params.train.eval else
                None,
                test=dataset_builder.get_pytorch_wrapper("test") if "test" in params.train.eval else None)

    # Init model
    model_cls = get_model(params)
    assert model_cls
    model = model_cls(params, datasets.train or datasets.test or datasets.valid)
    # model.summary()

    for parameter in model.parameters():
        logger.debug(parameter.shape)

    device_ids = [i for i in range(torch.cuda.device_count())]
    logger.info("GPUs: %s" % str(device_ids))
    model = DataParellelModel(model, device_ids)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info("Dataset: %s. Model: %s", str(dataset_builder), str(model_cls))
    if use_cuda:
        logger.info("CUDA available: %s", torch.cuda.get_device_name(0))

    # Load checkpoint or initialize new training
    if args.load:
        model.load_checkpoint(args.load)
        init_dirs(params)
        logger.info("Saved model loaded: %s", args.load)
        if mode == "train":
            logger.info("Epoch: %f", model.global_step / len(datasets.train))
    else:
        params.set('training_id', datetime.now().strftime('%Y%m%d-%H%M%S'))
        init_dirs(params)

    return params, args, model, datasets
