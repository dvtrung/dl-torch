import random

from dlex.utils import logger
from sklearn.model_selection import train_test_split


class SklearnDataset:
    def __init__(self, builder):
        self.builder = builder
        self.X = None
        self.y = None
        self.X_train = self.X_test = self.X_valid = None
        self.y_train = self.y_test = self.y_valid = None

    @property
    def configs(self):
        return self.params.dataset

    @property
    def params(self):
        return self.builder.params

    def init_dataset(self, X, y):
        if -1 in y:  # [-1, 1] -> [0, 1]
            y = [i if i != -1 else 0 for i in y]
        elif 0 not in y:  # [1, 2, 3] -> [0, 1, 2]
            y = [i - 1 for i in y]

        self.X, self.y = X, y

        if self.params.train.cross_validation:
            data = list(zip(X, y))
            random.seed(self.params.random_seed)
            random.shuffle(data)
            if False:  # split evenly
                classes = set(y)
                data_by_class = {c: [d for d in data if d[1] == c] for c in classes}
                data = []
                for i in range(max([len(d) for d in data_by_class.values()])):
                    for c in classes:
                        if i < len(data_by_class[c]):
                            data.append(data_by_class[c][i])

            X, y = zip(*data)

            logger.info("Initializing fold %d...", self.configs.cv_current_fold)

            pos_start = len(y) // self.configs.cv_num_folds * (self.configs.cv_current_fold - 1)
            pos_end = pos_start + len(y) // self.configs.cv_num_folds

            self.X_train = X[:pos_start] + X[pos_end:]
            self.X_test = X[pos_start:pos_end]
            self.y_train = y[:pos_start] + y[pos_end:]
            self.y_test = y[pos_start:pos_end]

            if self.configs.valid_set_ratio:
                num_valid = int(len(self.X_train) * self.configs.valid_set_ratio)
                self.X_valid = self.X_train[:num_valid]
                self.y_valid = self.y_train[:num_valid]
                self.X_train = self.X_train[num_valid:]
                self.y_train = self.y_train[num_valid:]
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(
                    X, y,
                    test_size=self.params.dataset.test_size or 0.2,
                    train_size=self.params.dataset.train_size)
