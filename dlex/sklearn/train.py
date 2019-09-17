"""Train a model."""
import importlib
from datetime import datetime

import numpy as np
from sklearn.model_selection import KFold

from dlex.configs import Configs
from dlex.utils.logging import logger, logging
from dlex.utils.model_utils import get_dataset
from dlex.utils.utils import init_dirs


def get_model(params):
    """Return the model class by its name."""
    module_name, class_name = params.model.name.rsplit('.', 1)
    i = importlib.import_module(module_name)
    return getattr(i, class_name)


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

    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.info("Dataset: %s. Model: %s", str(dataset_builder), str(model_cls))
    logger.info("Training started.")

    if params.train.cross_validation:
        scores = []
        cv = KFold(n_splits=10, random_state=42, shuffle=True)
        for train_index, test_index in cv.split(dataset.X):
            X_train, X_test, y_train, y_test = dataset.X[train_index], dataset.X[test_index], dataset.y[train_index], dataset.y[test_index]
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)
            logger.info("Fold score: %f", score)

        # model.fit(dataset.X_train, dataset.y_train)
        # scores = cross_val_score(model, dataset.X, dataset.y, cv=10)
        score = np.mean(scores)
        # score = model.score(dataset.X_test, dataset.y_test)
        logger.info("Score: %f", score)
        logger.info("Score deviation: %f", np.var(scores))
    else:
        model.fit(dataset.X_train, dataset.y_train)
        score = model.score(dataset.X_test, dataset.y_test)
        logger.info(score)


if __name__ == "__main__":
    main()
