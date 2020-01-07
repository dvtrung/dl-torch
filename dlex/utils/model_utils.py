"""Model utils"""
import importlib
import json
import os

from dlex.datasets.builder import DatasetBuilder


def get_dataset(params) -> DatasetBuilder:
    """Return the dataset class by its name."""
    if params.dataset.name:
        if '.' in params.dataset.name:
            module_name, class_name = params.dataset.name.rsplit('.', 1)
        else:
            module_name = f"src.datasets.{params.dataset.name.lower()}"
            class_name = params.dataset.name

    elif params.dataset.alias:
        from dlex.datasets.aliases import dataset_aliases
        module_name, class_name = dataset_aliases[params.dataset.alias].rsplit('.', 1)
    i = importlib.import_module(module_name)
    return getattr(i, class_name)(params)


def load_results(params):
    """Load all saved results at each checkpoint."""
    path = os.path.join(params.log_dir, "results.json")
    if os.path.exists(path):
        with open(os.path.join(params.log_dir, "results.json")) as f:
            return json.load(f)
    else:
        return {
            "best_results": {},
            "evaluations": []
        }


def add_result(params, new_result):
    """Add a checkpoint for evaluation result."""
    ret = load_results(params)
    ret["evaluations"].append(new_result)
    for m in params.test.metrics:
        if m not in ret["best_results"] or \
                new_result['result'][m] > ret['best_results'][m]['result'][m]:
            ret["best_results"][m] = new_result
    with open(os.path.join(params.log_dir, "results.json"), "w") as f:
        f.write(json.dumps(ret, indent=4))
    return ret["best_results"]