from sklearn.svm import SVC as _SVC

from dlex.configs import Params


class SVC(_SVC):
    def __init__(self, params: Params, dataset):
        super().__init__(
            gamma='scale',
            kernel=params.model.kernel or 'rbf')
        self.params = params

    def score(self, X, y, metric="acc"):
        if metric == "acc":
            return super().score(X, y) * 100

    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y, sample_weight)

    def predict(self, X):
        return super().predict(X)

