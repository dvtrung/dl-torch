from dlex import ModelReport
from dlex.configs import Configs


class FrameworkBackend:
    def __init__(
            self,
            argv=None,
            params=None,
            configs: Configs = None,
            training_idx: int = None,
            report_queue=None):
        self.params = params
        self.configs = configs
        self.argv = argv
        self.args = configs.args
        self.training_idx = training_idx

        report = ModelReport(training_idx)
        self.report = report
        self.report_queue = report_queue

    def run_train(self):
        raise NotImplemented

    def run_evaluate(self):
        raise NotImplemented