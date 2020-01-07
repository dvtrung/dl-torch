from dlex.configs import Configs
from dlex.utils.model_utils import get_dataset


def main(params=None, args=None):
    if not params and not args:
        configs = Configs(mode="train")
        envs, args = configs.environments, configs.args
        assert len(envs) == 1
        assert len(envs[0].configs_list) == 1
        params = envs[0].configs_list[0]

    dataset_builder = get_dataset(params)
    dataset_builder.prepare(download=args.download, preprocess=args.preprocess)
    assert dataset_builder, "Dataset not found."


if __name__ == "__main__":
    main()