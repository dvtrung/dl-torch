import json
import logging
import os

import numpy as np
from colorama import Fore, Style

logger = logging.getLogger('dlex')
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    level=logging.INFO,
    format=Fore.GREEN + '%(levelname)s - %(message)s' + Style.RESET_ALL)
logging.basicConfig(
    level=logging.DEBUG,
    format=Fore.MAGENTA + '%(levelname)s - %(message)s' + Style.RESET_ALL)
logging.basicConfig(
    level=logging.ERROR,
    format=Fore.RED + '%(levelname)s - %(message)s' + Style.RESET_ALL)
logging.basicConfig(
    level=logging.WARNING,
    format=Fore.RED + '%(levelname)s - %(message)s' + Style.RESET_ALL)

# Here we define our formatter
formatter = logging.Formatter(Fore.BLUE + '%(asctime)s - %(levelname)s - %(message)s' + Style.RESET_ALL)


epoch_info_logger = logging.getLogger('dlex-epoch-info')
epoch_info_logger.setLevel(logging.INFO)
epoch_step_info_logger = logging.getLogger('dlex-epoch-step-info')
epoch_step_info_logger.setLevel(logging.INFO)


def set_log_dir(params):
    os.makedirs(params.log_dir, exist_ok=True)

    log_info_handler = logging.FileHandler(
        os.path.join(params.log_dir, "info.log"))
    log_info_handler.setLevel(logging.INFO)
    log_info_handler.setFormatter(formatter)
    logger.addHandler(log_info_handler)

    log_debug_handler = logging.FileHandler(
        os.path.join(params.log_dir, "debug.log"))
    log_debug_handler.setLevel(logging.DEBUG)
    log_debug_handler.setFormatter(formatter)
    logger.addHandler(log_debug_handler)

    log_epoch_info_handler = logging.FileHandler(
        os.path.join(params.log_dir, "epoch-info.log"))
    log_epoch_info_handler.setLevel(logging.INFO)
    epoch_info_logger.addHandler(log_epoch_info_handler)

    log_epoch_step_info_handler = logging.FileHandler(
        os.path.join(params.log_dir, "epoch-step-info.log"))
    log_epoch_step_info_handler.setLevel(logging.INFO)
    epoch_step_info_logger.addHandler(log_epoch_step_info_handler)


def beautify(obj):
    if type(obj) is np.ndarray:
        return "[%s]" % ('\t'.join(["%.4f" % x for x in obj]))


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


def log_result(params, new_result, is_better_result):
    """Add a checkpoint for evaluation result.
    :return best result after adding new result
    """
    ret = load_results(params)
    ret["evaluations"].append(new_result)
    for m in params.test.metrics:
        if m not in ret["best_results"] or \
                is_better_result(m, ret['best_results'][m]['result'][m], new_result['result'][m]):
            ret["best_results"][m] = new_result
    with open(os.path.join(params.log_dir, "results.json"), "w") as f:
        f.write(json.dumps(ret, indent=4))
    return ret["best_results"]