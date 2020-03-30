import os
from typing import Tuple

from tensorflow.python.keras.optimizers import SGD
from tqdm import tqdm

from dlex.configs import Configs, Params, ModuleConfigs
from dlex.datasets.torch import Dataset
from dlex.tf.models import BaseModel
from dlex.utils.logging import logger
from dlex.utils.model_utils import add_result
from dlex.tf.utils.model_utils import get_model, get_dataset


def evaluate(
        model: BaseModel,
        dataset: Dataset,
        params: Params,
        save_result=False,
        output=False,
        summary_writer=None) -> Tuple[dict, dict, list]:

    total = {key: 0 for key in params.test.metrics}
    acc = {key: 0. for key in params.test.metrics}
    outputs = []
    for batch in tqdm(dataset.all(), desc="Eval"):
        y_pred, others = model.infer(batch)
        for key in params.test.metrics:
            _acc, _total = dataset.evaluate_batch(y_pred, batch, metric=key)
            acc[key] += _acc
            total[key] += _total
        if output:
            for i, predicted in enumerate(y_pred):
                str_input, str_ground_truth, str_predicted = dataset.format_output(
                    predicted, batch.item(i))
                outputs.append('\n'.join([str_input, str_ground_truth, str_predicted]))
        if summary_writer is not None:
            model.write_summary(summary_writer, batch, (y_pred, others))

    result = {
        "epoch": "%.1f" % model.epoch,
        "result": {key: acc[key] / total[key] for key in acc}
    }
    best_result = add_result(params, result) if save_result else None

    return result, best_result, outputs


def main(argv=None):
    """Main program."""
    configs = Configs(mode="eval", argv=argv)
    params, args = configs.params, configs.args

    dataset = get_dataset(params)
    dataset_test = dataset.get_keras_wrapper("validation")
    # Init model
    model_cls = get_model(params)
    assert model_cls
    model = model_cls(params, dataset_test).model

    model.compile(
        optimizer=SGD(0.1, momentum=0.9),
        loss="categorical_crossentropy",
        metrics=["acc"])

    # checkpoint
    checkpoint_path = os.path.join(configs.checkpoint_dir, "latest.h5")
    logger.info("Load checkpoint from %s" % checkpoint_path)
    model.load_weights(checkpoint_path)

    res = model.evaluate_generator(
        dataset_test.generator,
        steps=len(dataset_test) // params.train.batch_size,
        verbose=1
    )
    print(res)


if __name__ == "__main__":
    main()
