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
    params_list, args = configs.params_list, configs.args

    results = []
    for variables, params in params_list:
        results.append(train(params, args))

        print("--------- REPORT ---------")
        variables = list(params_list[0][0].keys())
        if len(variables) == 2 and len(results[0]) == 1:
            import pandas
            data = {}
            for i in range(len(params_list)):
                variable_values = params_list[i][0]
                val0 = variable_values[variables[0]]
                val1 = variable_values[variables[1]]
                if val0 not in data:
                    data[val0] = {}
                if val1 not in data[val0]:
                    if i < len(results):
                        data[val0][val1] = list(results[i].values())[0]
            print(pandas.DataFrame(data))
        else:
            for i in range(len(params_list)):
                variable_values = params_list[i][0]
                print("Configs: ", variable_values)
                print("Result: ", results[i])


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

    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.info("Dataset: %s. Model: %s", str(dataset_builder), str(model_cls))
    logger.info("Training started.")

    if params.train.cross_validation:
        scores = {}
        cv = KFold(n_splits=10, random_state=42, shuffle=True)
        for train_index, test_index in cv.split(dataset.X):
            X_train, X_test, y_train, y_test = dataset.X[train_index], dataset.X[test_index], dataset.y[train_index], dataset.y[test_index]
            model.fit(X_train, y_train)
            for metric in params.test.metrics:
                if metric not in scores:
                    scores[metric] = []
                scores[metric].append(model.score(X_test, y_test))

        # model.fit(dataset.X_train, dataset.y_train)
        # scores = cross_val_score(model, dataset.X, dataset.y, cv=10)
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
        report_callback(ret)
    return ret


if __name__ == "__main__":
    configs = Configs(mode="train", argv=None)
    params_list, args = configs.params_list, configs.args
    main()