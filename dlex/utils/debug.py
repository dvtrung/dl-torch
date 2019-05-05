import torch.nn as nn
from .logging import logger

class PrintLayer(nn.Module):
    def __init__(self, name=None):
        super(PrintLayer, self).__init__()
        self._name = name

    def forward(self, x, print_value=False):
        logger.debug("Tensor %s: Shape: %s", self._name, str(x.shape))
        if print_value:
            logger.debug(str(x))
        return x
