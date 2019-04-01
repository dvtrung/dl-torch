from configs import Configs
from utils.model_utils import get_model, get_dataset

if __name__ == "__main__":
    configs = Configs()
    configs.parse_args()
    print("Config path:", configs.args.config_path)
    configs.get_params()
    params = configs.params
    args = configs.args

    Dataset = get_dataset(configs.params)
    Model = get_model(configs.params)

    dataset = Dataset(params)
