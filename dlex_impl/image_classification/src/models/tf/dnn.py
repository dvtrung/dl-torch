from dlex import Params
from dlex.datasets.tf import Dataset
from dlex.tf import BaseModelV1


class DNN(BaseModelV1):
    def __init__(self, params: Params, dataset: Dataset):
        super().__init__(params, dataset)

