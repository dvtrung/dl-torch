from torchvision.transforms import transforms
from torchvision.datasets import MNIST as TorchMNIST

from dlex.datasets.image.torch import PytorchImageDataset
from dlex.torch import Batch
from dlex.torch.utils.ops_utils import maybe_cuda


class MNIST(PytorchImageDataset):
    """CIFAR10 dataset"""

    def __init__(self, builder, mode, params):
        super().__init__(builder, mode, params)
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.mnist = TorchMNIST(
            builder.get_working_dir(),
            train=mode == "train",
            transform=img_transform,
            download=True)

    def collate_fn(self, batch) -> Batch:
        ret = super().collate_fn(batch)
        return Batch(X=maybe_cuda(ret[0]), Y=maybe_cuda(ret[1]))

    @property
    def num_classes(self):
        return 10

    @property
    def num_channels(self):
        return 3

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        return self.mnist[idx]