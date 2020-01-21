import os
import random
import traceback
from typing import Tuple

import torch
from dlex.datatypes import ModelReport
from dlex.utils.utils import get_unused_gpus
from tqdm import tqdm

from dlex.configs import MainConfig, Configs
from dlex.datasets.torch import Dataset
from dlex.torch.models.base import BaseModel
from dlex.torch.utils.utils import load_model
from dlex.utils.logging import logger


def evaluate(
        model: BaseModel,
        dataset: Dataset,
        params: MainConfig,
        configs: Configs,
        output_path,
        report: ModelReport = None,
        tqdm_desc="Eval",
        tqdm_position=None) -> Tuple[dict, list]:
    """
    Evaluate model and save result.
    :param model:
    :param dataset:
    :param params:
    :param output_path: path without extension
    :param report:
    :return:
    """
    model.module.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        data_iter = dataset.get_iter(
            batch_size=params.test.batch_size or params.train.batch_size)

        # total = {key: 0 for key in params.test.metrics}
        # acc = {key: 0. for key in params.test.metrics}
        results = {metric: 0. for metric in params.test.metrics}
        outputs = []
        y_pred_all, y_ref_all, extra_all = [], [], []
        for batch in tqdm(
                data_iter,
                desc=tqdm_desc,
                leave=False,
                position=tqdm_position,
                disable=not configs.args.show_progress):
            try:
                if batch is None or len(batch) == 0:
                    raise Exception("Batch size 0")

                inference_outputs = model.infer(batch)
                y_pred, y_ref, others = inference_outputs[0], inference_outputs[1], inference_outputs[2:]

                y_pred_all += y_pred
                y_ref_all += y_ref
                # for metric in params.test.metrics:
                #     if metric == "loss":
                #         loss = model.get_loss(batch, model_output).item()
                #         _acc, _total = loss * len(y_pred), len(y_pred)
                #     else:
                #         _acc, _total = dataset.evaluate_batch(y_pred, batch, metric=metric)
                #     acc[metric] += _acc
                #     total[metric] += _total

                for i, predicted in enumerate(y_pred):
                    str_input, str_ground_truth, str_predicted = dataset.format_output(
                        predicted, batch.item(i))
                    outputs.append(dict(
                        input=str_input,
                        reference=str_ground_truth,
                        hypothesis=str_predicted))
                    # logger.info(outputs[-1])

                if report.summary_writer is not None:
                    model.write_summary(report.summary_writer, batch, (y_pred, others))
            except Exception:
                logger.error(traceback.format_exc())

        for metric in params.test.metrics:
            results[metric] = dataset.evaluate(y_pred_all, y_ref_all, metric, output_path)

    result = {
        "epoch": "%.1f" % model.current_epoch,
        "result": {key: results[key] for key in results}
    }

    return result, outputs


def main(params=None, configs: Configs = None, report=None):
    """Main program."""
    report = ModelReport()
    params, args, model, datasets = load_model("test", report, None, params, args)

    gpu = args.gpu or get_unused_gpus(args)
    params.gpu = gpu

    for mode in params.train.eval:
        result, outputs = evaluate(
            model, datasets.get_dataset(mode), params, configs,
            output_path=os.path.join(configs.log_dir, "results", f"{args.load}_{mode}"), report=report)

        for output in random.choices(outputs, k=50):
            logger.info(str(output))

    logger.info(str(result))


if __name__ == "__main__":
    main(None)
