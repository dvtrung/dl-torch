import random

from sklearn.model_selection import train_test_split


class SklearnDataset:
    def __init__(self, builder):
        self.builder = builder
        self.X = None
        self.y = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

    @property
    def configs(self):
        return self.params.dataset

    @property
    def params(self):
        return self.builder.params

    def init_dataset(self, X, y):
        self.X, self.y = X, y

        if self.params.dataset.cross_validation:
            data = list(zip(X, y))
            random.shuffle(data)
            X, y = zip(*data)
            pos_start = len(y) * (self.params.dataset.cv_current_fold - 1) // self.params.dataset.cross_validation
            pos_end = pos_start + len(y) // self.params.dataset.cross_validation
            self.X_train = X[:pos_start] + X[pos_end:]
            self.X_test = X[pos_start:pos_end]
            self.y_train = y[:pos_start] + y[pos_end:]
            self.y_test = y[pos_start:pos_end]
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(
                    X, y,
                    test_size=self.params.dataset.test_size or 0.2,
                    train_size=self.params.dataset.train_size)
