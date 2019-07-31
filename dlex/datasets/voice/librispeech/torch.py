import os

from dlex.datasets.voice.torch import PytorchVoiceDataset


class PytorchLibriSpeech(PytorchVoiceDataset):
    input_size = 120

    def __init__(self, builder, mode):
        super().__init__(
            builder, mode,
            vocab_path=os.path.join(builder.get_processed_data_dir(), "vocab", "%s.txt" % builder.params.dataset.unit))
        cfg = self.params.dataset

        is_debug = mode == "debug"
        if mode == "debug":
            mode = "train"

        with open(
                os.path.join(builder.get_processed_data_dir(), "%s_%s" % (cfg.unit, mode) + '.csv'),
                'r', encoding='utf-8') as f:
            lines = f.read().split('\n')[1:]
            lines = [l.split('\t') for l in lines if l != ""]
            self._data = [{
                'X_path': l[0],
                'Y': [int(w) for w in l[1].split(' ')],
            } for l in lines]