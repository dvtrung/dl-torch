from sklearn.svm import SVC as _SVC

from dlex.configs import MainConfig


class SVC(_SVC):
    def __init__(self, params: MainConfig, dataset):
        super().__init__(
            gamma='scale',
            kernel=params.model.kernel or 'rbf')

    def score(self, X, y, metric="acc"):
        if metric == "acc":
            return super().score(X, y) * 100
