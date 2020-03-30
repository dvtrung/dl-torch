from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Datasets:
    def __init__(
            self,
            backend: str,
            builder,
            train_set: str,
            valid_set: str,
            test_sets: List[str]):
        self.backend = backend
        self.builder = builder  # type: dlex.datasets.DatasetBuilder
        self._train = None
        self._valid = None
        self._tests = {}
        self._train_set = train_set
        self._valid_set = valid_set
        self._test_sets = test_sets

    @property
    def wrapper_fn(self):
        if self.backend == "tensorflow":
            return self.builder.get_tensorflow_wrapper
        elif self.backend == "pytorch":
            return self.builder.get_pytorch_wrapper

    @property
    def train_set(self):
        if self._train_set and not self._train:
            self._train = self.wrapper_fn(self._train_set)
        return self._train

    @property
    def valid_set(self):
        if self._valid_set and not self._valid:
            self._valid = self.wrapper_fn(self._valid_set)
        return self._valid

    @property
    def test_sets(self) -> Dict[str, Any]:
        if self._test_sets and not self._tests:
            self._tests = {ts: self.wrapper_fn(ts) for ts in self._test_sets}
        return self._tests