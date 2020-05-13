import abc
import pickle
from multiprocessing.queues import Queue
from typing import List

from dlex import ModelReport, Params, logger
from dlex.utils import set_seed


class FrameworkBackend:
    def __init__(
            self,
            params: Params = None,
            training_idx: int = None,
            report_queue: Queue = None):
        self.params = params
        self.configs = params.configs
        self.args = params.configs.args
        self.training_idx = training_idx

        report = ModelReport(training_idx)
        report.params = params
        params.training_idx = training_idx
        self.report = report
        self.report_queue = report_queue

        self.set_seed()

    @abc.abstractmethod
    def run_train(self):
        raise NotImplemented

    @abc.abstractmethod
    def run_evaluate(self):
        raise NotImplemented

    def update_report(self):
        if self.report_queue:
            # self.report_queue.put(self.report)
            self.report.save()

    def set_seed(self):
        set_seed(self.params.random_seed)