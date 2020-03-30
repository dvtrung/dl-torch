import logging

import tensorflow as tf
import tensorflow_federated as tff
from dlex import FrameworkBackend
from dlex.configs import Configs, Params
from dlex.tf.models.base_tff import TensorflowFederatedModel
from dlex.tf.utils.model_utils import get_model
from dlex.utils import Datasets, logger, set_seed
from dlex.utils.model_utils import get_dataset
from tqdm import tqdm


class TensorflowFederatedBackend(FrameworkBackend):
    def __init__(
            self,
            params: Params = None,
            training_idx: int = None,
            report_queue=None):
        super().__init__(params, training_idx, report_queue)

        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logger.info(f"Training started ({training_idx}).")

        self.report.results = {m: None for m in self.report.metrics}

    def run_train(self):
        set_seed(self.params.random_seed)
        self.train()

    def load_datasets(self, mode):
        """
        Load model and dataset
        :param mode: train, test, dev
        :return:
        """
        configs = self.configs
        report = self.report
        params = self.params
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
        return datasets

    def load_model(self, mode, datasets):
        params = self.params
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
        if self.args.load:
            self.configs.training_id = model.load_checkpoint(self.args.load)
            logger.info("Loaded checkpoint: %s", self.args.load)
            if mode == "train":
                logger.info("EPOCH: %f", model.global_step / len(datasets.train))

        return model

    def train(self):
        datasets = self.load_datasets("train")
        model_cls = get_model(self.params)

        def model_fn():
            model = model_cls(self.params, datasets.train)
            if isinstance(model, TensorflowFederatedModel):
                return model
            else:
                return tff.learning.from_keras_model(
                    model, datasets.train.dummy_batch,
                    loss=model.loss,
                    metrics=model.metrics)

        iterative_process = tff.learning.build_federated_averaging_process(
            model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

        state = iterative_process.initialize()
        for round in tqdm(range(self.params.train.num_epochs), desc="Round"):
            state, metrics = iterative_process.next(state, datasets.train.data)
            logger.info("Round %d: %s", round + 1, str(metrics))

        evaluation = tff.learning.build_federated_evaluation(model_fn)
        test_metrics = evaluation(state.model, datasets.test.data)
        logger.info(str(test_metrics))