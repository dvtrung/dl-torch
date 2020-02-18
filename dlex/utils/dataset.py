from dataclasses import dataclass


@dataclass
class Datasets:
    def __init__(self, backend, train=None, valid=None, test=None):
        self.backend = backend
        self.train = train
        self.valid = valid
        self.test = test

    def load_dataset(self, builder, mode):
        if self.backend == "tensorflow":
            fn = builder.get_tensorflow_wrapper
        elif self.backend == "pytorch":
            fn = builder.get_pytorch_wrapper

        if mode == "train":
            self.train = fn(mode)
        elif mode == "test":
            self.test = fn(mode)
        elif mode in {"valid", "dev"}:
            self.valid = fn(mode)

    def get_dataset(self, mode):
        if mode == "test":
            return self.test
        elif mode in {"valid", "dev"}:
            return self.valid