import torch
import torch.nn as nn
import torch.nn.functional as F


def nll_loss(batch, output):
    return F.nll_loss(output, batch.Y.view(-1))


def output_mean(batch, output):
    return torch.mean(output)


def mse_loss(batch, output):
    img = batch.X
    img = img.view(img.size(0), -1)
    criterion = nn.MSELoss()
    return criterion(output, img)
