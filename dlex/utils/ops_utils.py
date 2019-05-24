"""Operators utils"""
import torch

CUDA = torch.cuda.is_available()

LongTensor = torch.cuda.LongTensor if CUDA else torch.LongTensor
FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


def Tensor(*args):
    x = torch.Tensor(args)
    return x.cuda() if CUDA else x


def maybe_cuda(x):
    return x.cuda() if CUDA else x
