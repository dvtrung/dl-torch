from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np


@dataclass
class ModelReport:
    training_idx: int = None
    launch_time: datetime = None

    current_epoch: int = None
    num_epochs: int = None

    metrics: List[str] = None
    results: Dict[str, float] = None
    current_results: Dict[str, float] = None
    epoch_valid_results: List[Dict[str, float]] = None
    epoch_test_results: List[Dict[str, float]] = None
    epoch_losses: List[float] = None
    status: str = None
    num_params: int = None
    num_trainable_params: int = None
    param_details: str = None

    cv_num_folds: int = None
    cv_current_fold: int = None
    summary_writer = None

    def finish(self):
        self.status = "finished"

    def get_result_text(self, metric, full=False):
        """
        Get progress and current result in readable text format. Text includes:
            - Result / Average result ± variance
            - Current epoch
            - Current fold (k-fold cross validation)
        :param metric:
        :return:
        """
        if self.results and metric in self.results:
            res = self.results[metric] or self.current_results[metric]

            if res is None:
                result = "-"
            elif type(res) == float:
                result = "%.2f" % res
            elif type(res) == list and self.cv_num_folds:
                if full:
                    result = "[" + ", ".join(["%.2f" % r for r in res]) + "] -> " + "%.2f ± %.2f" % (np.mean(res), np.std(res))
                else:
                    result = "%.2f ± %.2f" % (np.mean(res), np.std(res))
            else:
                result = str(type(self.results[metric]))
            return result
        else:
            return "-"

    def get_status_text(self):
        if self.status == "finished":
            status = "done"
        elif self.cv_num_folds is not None:
            status = f"CV {self.cv_current_fold - 1}/{self.cv_num_folds}"
        else:
            pbar = get_progress_bar(10, (self.current_epoch - 1) / self.num_epochs)
            status = f"{pbar} {self.current_epoch - 1}/{self.num_epochs}"
        return status


def get_progress_bar(width, percent):
    progress = int(width * percent)
    return "[%s%s]" % ('#' * progress, ' ' * (width - progress))