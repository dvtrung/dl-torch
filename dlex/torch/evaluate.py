from typing import Tuple

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from dlex.datasets.torch import PytorchDataset
from dlex.configs import Configs, AttrDict
from dlex.utils.model_utils import get_dataset
from dlex.torch.utils.model_utils import get_model, load_checkpoint
from dlex.utils.logging import logger
from dlex.utils.utils import init_dirs
from dlex.torch.models.base import DataParellelModel


def evaluate(
        model,
        dataset: PytorchDataset,
        params: AttrDict,
        output=False,
        summary_writer=None) -> Tuple[dict, list]:
    """
    Evaluate model and save result.
    :type model: dlex.models.base.BaseModel
    :type dataset: dlex.datasets.base.BaseDataset
    :type params: dlex.configs.AttrDict
    :type save_result: bool
    :type output: bool
    :type summary_writer: torch.utils.tensorboard.SummaryWriter | None
    :return: result
    :rtype:
    :return: best_result
    :rtype:
    :return: outputs
    :rtype: list[str]
    """
    data_loader = DataLoader(
        dataset,
        batch_size=params.test.batch_size or params.train.batch_size,
        collate_fn=dataset.collate_fn)

    total = {key: 0 for key in params.test.metrics}
    acc = {key: 0. for key in params.test.metrics}
    outputs = []
    for batch in tqdm(data_loader, desc="Eval"):
        try:
            y_pred, others = model.infer(batch)
            print(model.module.infer_log(batch, y_pred, params.verbose))
            for key in params.test.metrics:
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
                    print(outputs[-1])
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
            print("---")
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
    logger.info("Training on %s" % str(device_ids))
    model = DataParellelModel(model, device_ids)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
        model.cuda()

    if args.load is None:
        raise Exception("A saved model file must be specified.")
    load_checkpoint(args.load, params, model)
    init_dirs(params)

    logger.info("Saved model loaded: %s", args.load)

    result, outputs = evaluate(model, dataset_test, params, output=True)

    for output in outputs:
        logger.info(str(output))
        logger.info("---")

    logger.info(str(result))


if __name__ == "__main__":
    main()
