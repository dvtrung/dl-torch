import torch

CUDA = torch.cuda.is_available()

def Tensor(*args):
    x = torch.Tensor(args)
    return x.cuda() if CUDA else x

def LongTensor(*args):
    x = torch.LongTensor(args)
    return x.cuda() if CUDA else x

def maybe_cuda(x):
    return x.cuda() if CUDA else x
