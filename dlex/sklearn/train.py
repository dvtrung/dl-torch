"""Train a model."""
import random
from dataclasses import dataclass
from datetime import datetime

from dlex.configs import Configs, AttrDict
from dlex.datasets.torch import PytorchDataset
from dlex.torch.evaluate import evaluate
from dlex.torch.models.base import DataParellelModel
from dlex.torch.utils.model_utils import get_model
from dlex.utils.logging import logger, epoch_info_logger, logging, log_result, json_dumps, \
    log_outputs
from dlex.utils.model_utils import get_dataset
from dlex.utils.utils import init_dirs


# from torch.utils.tensorboard import SummaryWriter


@dataclass
class Datasets:
    train: PytorchDataset
    valid: PytorchDataset = None
    test: PytorchDataset = None


def train(
        params: AttrDict,
        args,
        model: DataParellelModel,
        datasets: Datasets,
        summary_writer):
    epoch = model.global_step // len(datasets.train)
    num_samples = model.global_step % len(datasets.train)
    # num_samples = 0
    for current_epoch in range(epoch + 1, epoch + params.train.num_epochs + 1):
        log_dict = dict(epoch=current_epoch)
        log_dict['total_time'], log_dict['loss'] = train_epoch(
            current_epoch, params, args,
            model, datasets, summary_writer, num_samples)
        num_samples = 0

        def _evaluate(mode):
            # Evaluate model
            result, outputs = evaluate(
                model,
                getattr(datasets, mode),
                params,
                output=True,
                summary_writer=summary_writer)
            best_result = log_result(mode, params, result, datasets.train.builder.is_better_result)
            for metric in best_result:
                if best_result[metric] == result:
                    model.save_checkpoint(
                        "best" if len(params.test.metrics) == 1 else "%s-best-%s" % (mode, metric))
                    logger.info("Best %s for %s set reached: %f", metric, mode, result['result'][metric])
            return result, best_result, outputs

        test_result, _, test_outputs = _evaluate("test")
        log_outputs("test", params, test_outputs)
        log_dict['test_result'] = test_result['result']
        if datasets.valid is not None:
            valid_result, valid_best_result, valid_outputs = _evaluate("valid")
            log_outputs("valid", params, valid_outputs)
            log_dict['valid_result'] = valid_result['result']
            for metric in valid_best_result:
                if valid_best_result[metric] == valid_result:
                    logger.info("Best result: %f", test_result['result'][metric])
                    log_result(f"valid_test_{metric}", params, test_result, datasets.train.builder.is_better_result)
                    log_outputs("valid_test", params, test_outputs)

        for metric in test_result['result']:
            if summary_writer is not None:
                summary_writer.add_scalar("eval_%s" % metric, test_result['result'][metric], current_epoch)

        logger.info("Random samples")
        for output in random.choices(test_outputs, k=5):
            logger.info(str(output))

        epoch_info_logger.info(json_dumps(log_dict))
        logger.info(json_dumps(log_dict))


def main(argv=None):
    configs = Configs(mode="train", argv=argv)
    params, args = configs.params, configs.args

    # Init dataset
    dataset_builder = get_dataset(params)
    assert dataset_builder
    if not args.no_prepare:
        dataset_builder.prepare(download=args.download, preprocess=args.preprocess)

    dataset = dataset_builder.get_sklearn_wrapper("train")

    # Init model
    model_cls = get_model(params)
    assert model_cls
    model = model_cls(params)

    # Load checkpoint or initialize new training
    if args.load:
        model.load_checkpoint(args.load)
        init_dirs(params)
        logger.info("Saved model loaded: %s", args.load)
        logger.info("Epoch: %f", model.global_step / len(dataset.X_train))
    else:
        params.set('training_id', datetime.now().strftime('%Y%m%d-%H%M%S'))
        init_dirs(params)

    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.info("Dataset: %s. Model: %s", str(dataset_builder), str(model_cls))
    logger.info("Training started.")

    logger.info("Training started")
    model.fit(dataset.X_train, dataset.y_train)
    logger.info("Score: %f", model.score(dataset.X_test, dataset.y_test))


if __name__ == "__main__":
    main()
