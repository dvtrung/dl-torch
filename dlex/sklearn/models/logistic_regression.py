from sklearn.linear_model import LogisticRegression as _LogisticRegression

from dlex.configs import AttrDict


class LogisticRegression(_LogisticRegression):
    def __init__(self, params: AttrDict, dataset):
        super().__init__()