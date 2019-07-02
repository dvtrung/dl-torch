import os

from dlex.datasets.voice.torch import PytorchVoiceDataset


class PytorchVIVOS(PytorchVoiceDataset):
    def __init__(self, builder, mode, params):
        cfg = params.dataset
        super().__init__(builder, mode, params, builder.get_vocab_path(cfg.unit))

        is_debug = mode == "debug"
        if mode == "debug":
            mode = "train"

        self._data = self.load_data(os.path.join(builder.get_processed_data_dir(), "%s_%s" % (cfg.unit, mode) + '.csv'))
        if is_debug:
            self._data = self._data[:cfg.debug_size]

    @property
    def input_size(self):
        return self.params.dataset.feature.num_filters