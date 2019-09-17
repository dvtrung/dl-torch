import random
from typing import Tuple

import torch
from tqdm import tqdm

from dlex.configs import AttrDict
from dlex.datasets.torch import PytorchDataset
from dlex.torch.models.base import BaseModel
from dlex.torch.utils.utils import load_model
from dlex.utils.logging import logger


def evaluate(
        model: BaseModel,
        dataset: PytorchDataset,
        params: AttrDict,
        output=False,
        summary_writer=None) -> Tuple[dict, list]:
    """Evaluate model and save result."""
    model.module.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        data_iter = dataset.get_iter(
            batch_size=params.test.batch_size or params.train.batch_size)

        # total = {key: 0 for key in params.test.metrics}
        # acc = {key: 0. for key in params.test.metrics}
        results = {metric: 0. for metric in params.test.metrics}
        outputs = []
        y_pred_all, y_ref_all = [], []
        for batch in tqdm(data_iter, desc="Eval"):
            try:
                if batch is None or batch.X.shape[0] == 0:
                    raise Exception("Batch size 0")
                y_pred, y_ref, model_output, others = model.infer(batch)
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
                if output:
                    for i, predicted in enumerate(y_pred):
                        str_input, str_ground_truth, str_predicted = dataset.format_output(
                            predicted, batch.item(i))
                        outputs.append(dict(
                            input=str_input,
                            reference=str_ground_truth,
                            hypothesis=str_predicted))
                        # print(outputs[-1])
                if summary_writer is not None:
                    model.write_summary(summary_writer, batch, (y_pred, others))
            except RuntimeError as e:
                import gc
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                            print(type(obj), obj.size() if hasattr(obj, 'size') else "")
                    except e:
                        pass
                logger.error(str(e))
            except Exception as e:
                logger.error(str(e))

        for metric in params.test.metrics:
            results[metric] = dataset.evaluate(y_pred_all, y_ref_all, metric)

    result = {
        "epoch": "%.1f" % model.current_epoch,
        "result": {key: results[key] for key in results}
    }

    return result, outputs


def main():
    """Main program."""
    params, args, model, datasets = load_model("test")
    #torch.manual_seed(params.seed)
    result, outputs = evaluate(model, datasets.test, params, output=True)

    for output in random.choices(outputs, k=50):
        logger.info(str(output))

    logger.info(str(result))


if __name__ == "__main__":
    main()
