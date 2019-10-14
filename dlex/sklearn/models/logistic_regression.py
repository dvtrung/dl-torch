from sklearn.linear_model import LogisticRegression as _LogisticRegression

from dlex.configs import MainConfig


class LogisticRegression(_LogisticRegression):
    def __init__(self, params: MainConfig, dataset):
        super().__init__()