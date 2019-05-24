"""Download and preprocess dataset with configs."""

from .configs import Configs
from .utils.model_utils import get_dataset
from .utils.logging import logger


def main():
    """Read config and train model."""
    configs = Configs(mode="train")
    dataset_cls = get_dataset(configs.params)
    logger.info("Dataset: %s (%s)", configs.params.dataset.name, str(dataset_cls))

    # Init dataset
    dataset_cls.prepare(download=False, preprocess=True)


if __name__ == "__main__":
    main()
