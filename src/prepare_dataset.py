"""Train a model."""

from configs import Configs
from utils.model_utils import get_dataset


def main():
    """Read config and train model."""
    configs = Configs(mode="train")
    dataset_cls = get_dataset(configs.params)
    # Init dataset
    dataset_cls.prepare(force=configs.args.force_preprocessing)


if __name__ == "__main__":
    main()
