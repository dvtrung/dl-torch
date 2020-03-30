from dlex.datasets.torch import Dataset
from dlex.torch import Batch


class PytorchImageDataset(Dataset):
    def __init__(self, builder, mode):
        super().__init__(builder, mode)

    def collate_fn(self, batch) -> Batch:
        ret = super().collate_fn(batch)
        return Batch(X=self.maybe_cuda(ret[0]), Y=self.maybe_cuda(ret[1]))

    @property
    def num_channels(self):
        return self.builder.num_channels

    @property
    def input_shape(self):
        return self.builder.input_shape