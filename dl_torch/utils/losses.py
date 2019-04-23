import torch
import torch.nn as nn
import torch.nn.functional as F


def nll_loss(batch, output):
    y = batch['Y']
    return F.nll_loss(output, y.view(-1))


def seq_nll_loss(batch, output):
    y = batch['Y']
    decoder_outputs, _, other = output
    seqlist = other['sequence']
    loss = 0
    for step, step_output in enumerate(decoder_outputs):
        target = y[:, step + 1]
        loss += F.nll_loss(step_output.view(y.size(0), -1), target)

    return loss / len(decoder_outputs)


def output_mean(batch, output):
    return torch.mean(output)


def mse_loss(batch, output):
    img = batch['X']
    img = img.view(img.size(0), -1)
    criterion = nn.MSELoss()
    return criterion(output, img)
