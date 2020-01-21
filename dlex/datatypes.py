from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ModelReport:
    current_epoch: int = None
    num_epochs: int = None

    metrics: List[str] = None
    results: Dict[str, float] = None
    epoch_valid_results: List[Dict[str, float]] = None
    epoch_test_results: List[Dict[str, float]] = None
    epoch_losses: List[float] = None
    finished: bool = False
    num_params: int = None
    num_trainable_params: int = None
    param_details: str = None

    cv_num_folds: int = None
    cv_current_fold: int = None
    summary_writer = None