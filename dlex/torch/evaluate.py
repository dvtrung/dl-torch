import random
from typing import Tuple

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from dlex.datasets.torch import PytorchDataset
from dlex.configs import Configs, AttrDict
from dlex.utils.model_utils import get_dataset
from dlex.torch.utils.model_utils import get_model
from dlex.utils.logging import logger
from dlex.utils.utils import init_dirs
from dlex.torch.models.base import DataParellelModel, BaseModel


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

        total = {key: 0 for key in params.test.metrics}
        acc = {key: 0. for key in params.test.metrics}
        outputs = []
        for batch in tqdm(data_iter, desc="Eval"):
            try:
                if batch.X.shape[0] == 0:
                    raise Exception("Batch size 0")
                y_pred, model_output, others = model.infer(batch)
                # print(model.module.infer_log(batch, y_pred, params.verbose))
                for key in params.test.metrics:
                    if key == "loss":
                        loss = model.get_loss(batch, model_output).item()
                        _acc, _total = loss * len(y_pred), len(y_pred)
                    else:
                        _acc, _total = dataset.evaluate_batch(y_pred, batch, metric=key)
                    acc[key] += _acc
                    total[key] += _total
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

    result = {
        "epoch": "%.1f" % model.current_epoch,
        "result": {key: acc[key] / total[key] for key in acc}
    }

    return result, outputs


def main():
    """Main program."""
    configs = Configs(mode="eval")
    configs.parse_args()
    configs.get_params()
    params = configs.params
    args = configs.args

    torch.manual_seed(params.seed)

    dataset_builder = get_dataset(params)
    model_cls = get_model(params)

    # Init dataset
    if not args.no_prepare:
        dataset_builder.prepare()
    dataset_test = dataset_builder.get_pytorch_wrapper("test")

    # Init model
    model = model_cls(params, dataset_test)
    device_ids = [i for i in range(torch.cuda.device_count())]
    logger.info("Evaluating on %s" % str(device_ids))
    model = DataParellelModel(model, device_ids)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
        model.cuda()

    if args.load is None:
        raise Exception("A saved model file must be specified.")
    model.load_checkpoint(args.load)
    init_dirs(params)

    logger.info("Saved model loaded: %s", args.load)

    result, outputs = evaluate(model, dataset_test, params, output=True)

    for output in random.choices(outputs, k=50):
        logger.info(str(output))

    logger.info(str(result))


if __name__ == "__main__":
    main()
