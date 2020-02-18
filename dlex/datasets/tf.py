class Dataset:
    def __init__(self, builder, mode: str):
        self.params = builder.params
        self.mode = mode
        self._builder = builder
        self._data = None
        self._sampler = None

    @property
    def builder(self):
        return self._builder

    @property
    def data(self):
        return self._data

    @property
    def processed_data_dir(self) -> str:
        return self.builder.get_processed_data_dir()

    @property
    def raw_data_dir(self) -> str:
        return self.builder.get_raw_data_dir()

    @property
    def configs(self):
        return self.params.dataset

    def __len__(self):
        return len(self.data)