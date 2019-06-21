class KerasDataset:
    def __init__(self, dataset, mode, params):
        self._params = params
        self._mode = mode
        self._dataset = dataset

    @property
    def generator(self):
        return self._dataset.__iter__()