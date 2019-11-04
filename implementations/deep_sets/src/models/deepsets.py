from torch.utils.data import Dataset

from dlex import MainConfig
from dlex.torch.models import BaseModel


class DeepSets(BaseModel):
    def __init__(self, params: MainConfig, dataset: Dataset):
        super().__init__(params, dataset)