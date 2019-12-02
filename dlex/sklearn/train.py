"""Train a model."""
import importlib
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.model_selection import KFold

from dlex.configs import Configs
from dlex.utils.logging import logger, logging
from dlex.utils.model_utils import get_dataset
from dlex.utils.utils import init_dirs
from tqdm import tqdm


def get_model(params):
    """Return the model class by its name."""
    module_name, class_name = params.model.name.rsplit('.', 1)
    i = importlib.import_module(module_name)
    return getattr(i, class_name)


def main(argv=None):
    configs = Configs(mode="train", argv=argv)
    params_list, args = configs.params_list, configs.args

    results = []
    for variables, params in params_list:
        results.append(train(params, args))


def train(params, args, report_callback=None):
    # Init dataset
    dataset_builder = get_dataset(params)
    assert dataset_builder
    if not args.no_prepare:
        dataset_builder.prepare(download=args.download, preprocess=args.preprocess)

    dataset = dataset_builder.get_sklearn_wrapper("train")
    assert dataset

    # Init model
    model_cls = get_model(params)
    assert model_cls
    model = model_cls(params, dataset)

    # Load checkpoint or initialize new training
    if args.load:
        model.load_checkpoint(args.load)
        init_dirs(params)
        logger.info("Saved model loaded: %s", args.load)
        logger.info("Epoch: %f", model.global_step / len(dataset.X_train))
    else:
        params.set('training_id', datetime.now().strftime('%Y%m%d-%H%M%S'))
        init_dirs(params)

    logger.info("Dataset: %s. Model: %s", str(dataset_builder), str(model_cls))
    logger.info("Training started.")

    if params.train.cross_validation:
        scores = defaultdict(list)
        cv = KFold(n_splits=params.train.cross_validation, random_state=42, shuffle=True)
        for train_index, test_index in tqdm(
                cv.split(dataset.X), desc="Cross Validation", leave=True,
                total=params.train.cross_validation):
            X_train, X_test = dataset.X[train_index], dataset.X[test_index]
            y_train, y_test = dataset.y[train_index], dataset.y[test_index]
            model.fit(X_train, y_train)
            for metric in params.test.metrics:
                scores[metric].append(model.score(X_test, y_test))

        ret = {}
        for metric in scores:
            ret[metric] = np.mean(scores[metric])
            logger.info("Score (%s): %f", metric, ret[metric])
            logger.info("Score deviation (%s): %f", metric, np.var(scores[metric]))
    else:
        model.fit(dataset.X_train, dataset.y_train)
        ret = {}
        for metric in params.test.metrics:
            ret[metric] = model.score(dataset.X_test, dataset.y_test, metric or "f1")
        logger.info(ret)

    if report_callback:
        report_callback(ret, True)
    return ret


if __name__ == "__main__":
    configs = Configs(mode="train", argv=None)
    params_list, args = configs.params_list, configs.args
    main()