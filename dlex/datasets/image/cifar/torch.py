from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10 as TorchCIFAR10

from dlex.datasets.torch import PytorchDataset
from dlex.torch import Batch
from dlex.torch.utils.ops_utils import maybe_cuda


class CIFAR10(PytorchDataset):
    """CIFAR10 dataset"""

    def __init__(self, builder, mode, params):
        super().__init__(builder, mode, params)
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        self.cifar = TorchCIFAR10(
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
        return len(self.cifar)

    def __getitem__(self, idx):
        return self.cifar[idx]