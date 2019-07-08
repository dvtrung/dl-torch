"""Operators utils"""
import torch

CUDA = torch.cuda.is_available()
gpus = [i for i in range(torch.cuda.device_count())]
if len(gpus) > 0:
    torch.cuda.set_device(gpus[0])


LongTensor = torch.cuda.LongTensor if CUDA else torch.LongTensor
FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


def Tensor(*args):
    x = torch.Tensor(args)
    return x.cuda(device=gpus[0]) if CUDA else x


def maybe_cuda(x):
    return x.cuda() if CUDA else x
