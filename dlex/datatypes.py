from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ModelReport:
    metrics: List[str] = None
    results: Dict[str, float] = None
    finished: bool = False
    num_params: int = None
    num_trainable_params: int = None
    param_details: str = None

    cross_validation_total: int = None
    cross_validation_fold: int = None
    summary_writer = None