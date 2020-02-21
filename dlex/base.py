import abc
from multiprocessing.queues import Queue

from dlex import ModelReport
from dlex.configs import Configs


class FrameworkBackend:
    def __init__(
            self,
            argv=None,
            params=None,
            configs: Configs = None,
            training_idx: int = None,
            report_queue: Queue = None):
        self.params = params
        self.configs = configs
        self.argv = argv
        self.args = configs.args
        self.training_idx = training_idx

        report = ModelReport(training_idx)
        self.report = report
        self.report_queue = report_queue

        self.set_seed(params.random_seed)

    @abc.abstractmethod
    def run_train(self):
        raise NotImplemented

    @abc.abstractmethod
    def run_evaluate(self):
        raise NotImplemented

    def update_report(self):
        if self.report_queue:
            self.report_queue.put(self.report)

    @abc.abstractmethod
    def set_seed(self, seed):
        raise NotImplemented