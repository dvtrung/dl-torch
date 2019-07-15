from tensorflow.python import keras


class KerasDataset:
    def __init__(self, builder, mode, params):
        self.params = params
        self.mode = mode
        self.builder = builder
        self.dataset = None

    @property
    def dataset(self):
        return self.dataset

    @property
    def generator(self):
        return self.dataset.__iter__()

    def get_metrics(self):
        ls = list()
        for metric in self.params.test.metrics:
            if metric in ["acc"]:
                ls.append(metric)
            elif metric == 'top5':
                ls.append(lambda y_true, y_pred:
                          keras.metrics.top_k_categorical_accuracy(y_true, y_pred, 5))
        return ls