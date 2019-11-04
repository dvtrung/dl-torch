from sklearn.model_selection import train_test_split


class SklearnDataset:
    def __init__(self, builder):
        self.builder = builder

    @property
    def configs(self):
        return self.params.dataset

    @property
    def params(self):
        return self.builder.params

    def init_dataset(self, X, y):
        self.X, self.y = X, y
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(
                X, y,
                test_size=self.params.dataset.test_size or 0.2,
                train_size=self.params.dataset.train_size,
                random_state=self.params.dataset.random_state or 42)
