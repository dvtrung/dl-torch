from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from configs import Configs
from utils.model_utils import get_dataset, get_model, get_optimizer, \
    add_result, load_checkpoint
from utils.logging import logger
from utils.utils import init_dirs

def infer(model, dataset, params, save_result=False, output=False):
    results = []

    data_loader = DataLoader(
        dataset,
        batch_size=params.test_batch_size or params.batch_size,
        collate_fn=dataset.collate_fn)

    for batch in tqdm(data_loader, desc="Infer"):
        y_pred = model.infer(batch)
        for y, inp in zip(y_pred, batch):
            logger.info(dataset.format_output(y, inp, display="word+tag"))
        # logger.info('\n'.join([str(r) for r in ret]))

if __name__ == "__main__":
    configs = Configs(mode="infer")
    params = configs.params
    args = configs.args

    Dataset = get_dataset(params)
    Model = get_model(params)

    # Init dataset
    dataset_infer = Dataset("infer", params, args)

    # Init model
    model = Model(params, dataset_infer)
    if torch.cuda.is_available():
        logger.info("Cuda available: " + torch.cuda.get_device_name(0))
        model.cuda()

    if args.load is None:
        raise Exception("A saved model file must be specified.")
    load_checkpoint(args.load, params, model, None)
    init_dirs(params)
    torch.manual_seed(params.seed)

    infer(model, dataset_infer, params, output=True)
