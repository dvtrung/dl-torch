from sklearn.svm import SVC as _SVC

from dlex.configs import AttrDict


class SVC(_SVC):
    def __init__(self, params: AttrDict):
        super().__init__()