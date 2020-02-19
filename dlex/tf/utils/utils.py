import tensorflow as tf
from dlex import MainConfig
from dlex.configs import Configs
from dlex.datatypes import ModelReport
from dlex.tf.utils.model_utils import get_model
from dlex.utils import Datasets, logger, table2str
from dlex.utils.model_utils import get_dataset


def load_model(mode, report: ModelReport, argv=None, params: MainConfig = None, configs=None):
    """
    Load model and dataset
    :param mode: train, test, dev
    :param report:
    :param argv:
    :param params: if None, configs will be read from file
    :param args:
    :return:
    """

    if not configs:
        configs = Configs(mode=mode, argv=argv)
        envs, args = configs.environments, configs.args
        assert len(envs) == 1
        assert len(envs[0].configs_list) == 1
        params = envs[0].configs_list[0]
    else:
        args = configs.args

    report.metrics = params.test.metrics

    # Init dataset
    dataset_builder = get_dataset(params)
    assert dataset_builder, "Dataset not found."
    if not args.no_prepare:
        dataset_builder.prepare(download=args.download, preprocess=args.preprocess)
    if mode == "test":
        datasets = Datasets("tensorflow")
        for mode in params.train.eval:
            datasets.load_dataset(dataset_builder, mode)
    elif mode == "train":
        if args.debug:
            datasets = Datasets(
                "tensorflow",
                train=dataset_builder.get_tensorflow_wrapper("test"),
                test=dataset_builder.get_tensorflow_wrapper("test"))
        else:
            datasets = Datasets(
                "tensorflow",
                train=dataset_builder.get_tensorflow_wrapper("train"),
                valid=dataset_builder.get_tensorflow_wrapper("valid") if "valid" in params.train.eval else
                dataset_builder.get_tensorflow_wrapper("dev") if "dev" in params.train.eval else None,
                test=dataset_builder.get_tensorflow_wrapper("test") if "test" in params.train.eval else None)

    # Init model
    model_cls = get_model(params)
    assert model_cls, "Model not found."
    model = model_cls(params, datasets.train if datasets.train is not None else datasets.test or datasets.valid)
    # model.summary()

    # log model summary
    # parameter_details = [["Name", "Shape", "Trainable"]]
    # num_params = 0
    # num_trainable_params = 0
    # for n in tf.get_default_graph().as_graph_def().node:
    #     parameter_details.append([
    #         n.name,
    #         "test",
    #         "✓" if False else ""])
    #     num_params += np.prod(list(parameter.shape))
    #     if parameter.requires_grad:
    #         num_trainable_params += np.prod(list(parameter.shape))

    # s = table2str(parameter_details)
    # logger.debug(f"Model parameters\n{s}")
    # logger.debug(" - ".join([
    #     f"No. parameters: {num_params:,}",
    #     f"No. trainable parameters: {num_trainable_params:,}"
    # ]))
    # report.param_details = s
    # report.num_params = num_params
    # report.num_trainable_params = num_trainable_params

    # use_cuda = torch.cuda.is_available()
    # if use_cuda and params.gpu:
    #     gpus = [f"cuda:{g}" for g in params.gpu]
    #     model = DataParellelModel(model, gpus)
    #     logger.info("Start training using %d GPU(s): %s", len(params.gpu), str(params.gpu))
    #     torch.cuda.set_device(torch.device(gpus[0]))
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     model.to(gpus[0])
    # else:
    #     model = DataParellelModel(model, ['cpu'])

    # logger.debug("Dataset: %s. Model: %s", str(dataset_builder), str(model_cls))
    # if use_cuda:
    #     logger.info("CUDA available: %s", torch.cuda.get_device_name(0))

    # Load checkpoint or initialize new training
    if args.load:
        configs.training_id = model.load_checkpoint(args.load)
        logger.info("Loaded checkpoint: %s", args.load)
        if mode == "train":
            logger.info("EPOCH: %f", model.global_step / len(datasets.train))

    return params, args, model, datasets