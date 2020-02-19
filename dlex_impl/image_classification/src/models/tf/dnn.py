from dlex import MainConfig
from dlex.datasets.tf import Dataset
from dlex.tf import BaseModel_v1


class DNN(BaseModel_v1):
    def __init__(self, params: MainConfig, dataset: Dataset):
        super().__init__(params, dataset)

