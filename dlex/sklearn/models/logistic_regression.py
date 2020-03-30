from sklearn.linear_model import LogisticRegression as _LogisticRegression

from dlex.configs import Params


class LogisticRegression(_LogisticRegression):
    def __init__(self, params: Params, dataset):
        super().__init__()