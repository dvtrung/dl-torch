import json
import logging
import os

import numpy as np
from tqdm import tqdm


import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class DebugFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        logging.FileHandler.__init__(self, filename, mode, encoding, delay)

    def emit(self, record):
        if not record.levelno == logging.DEBUG:
            return
        logging.FileHandler.emit(self, record)


class ErrorFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        logging.FileHandler.__init__(self, filename, mode, encoding, delay)

    def emit(self, record):
        if not record.levelno == logging.ERROR:
            return
        logging.FileHandler.emit(self, record)


log_format = '%(asctime)s - %(levelname)s - %(message)s'
bold_seq = '\033[1m'
#colorlog.basicConfig(format=(
#    f'{bold_seq} '
#    '%(log_color)s '
#    f'{log_format}'
#))

logger = logging.getLogger('dlex')
logger.setLevel(logging.DEBUG)


def set_log_level(level: str):
    level = "error"
    level = dict(
        none=logging.NOTSET,
        info=logging.INFO,
        warn=logging.WARN,
        error=logging.ERROR,
        debug=logging.DEBUG)[level]
    logger.setLevel(level)


epoch_info_logger = logging.getLogger('dlex-epoch-info')
epoch_info_logger.setLevel(logging.INFO)
epoch_info_logger.propagate = False
epoch_step_info_logger = logging.getLogger('dlex-epoch-step-info')
epoch_step_info_logger.setLevel(logging.INFO)
epoch_step_info_logger.propagate = False


def set_log_dir(configs):
    os.makedirs(configs.log_dir, exist_ok=True)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    sym_path = os.path.abspath(os.path.join(configs.log_dir, os.pardir, "latest"))
    if os.path.exists(sym_path):
        os.unlink(sym_path)
    # TODO: symlink doesn't work correctly
    # os.symlink(params.log_dir, sym_path, True)

    log_info_handler = logging.FileHandler(os.path.join(configs.log_dir, "info.log"))
    log_info_handler.setLevel(logging.INFO)
    log_info_handler.setFormatter(formatter)
    logger.addHandler(log_info_handler)

    log_debug_handler = DebugFileHandler(os.path.join(configs.log_dir, "debug.log"))
    log_debug_handler.setLevel(logging.DEBUG)
    log_debug_handler.setFormatter(formatter)
    logger.addHandler(log_debug_handler)

    log_error_handler = ErrorFileHandler(os.path.join(configs.log_dir, "error.log"))
    log_error_handler.setLevel(logging.ERROR)
    log_error_handler.setFormatter(formatter)
    logger.addHandler(log_error_handler)

    if configs.args.show_logs:
        logger.addHandler(TqdmLoggingHandler())


    # log_epoch_info_handler = logging.FileHandler(
    #     os.path.join(configs.log_dir, "epoch-info.log"))
    # log_epoch_info_handler.setLevel(logging.INFO)
    # epoch_info_logger.addHandler(log_epoch_info_handler)

    # log_epoch_step_info_handler = logging.FileHandler(
    #     os.path.join(configs.log_dir, "epoch-step-info.log"))
    # log_epoch_step_info_handler.setLevel(logging.INFO)
    # epoch_step_info_logger.addHandler(log_epoch_step_info_handler)


def beautify(obj):
    if type(obj) is np.ndarray:
        return "[%s]" % ('\t'.join(["%.4f" % x for x in obj]))


def load_results(mode, params):
    """Load all saved results at each checkpoint."""
    path = os.path.join(params.log_dir, "results_%s.json" % mode)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    else:
        return {
            "best_results": {},
            "evaluations": []
        }


def log_result(mode: str, params, new_result: float, is_better_result):
    """Add a checkpoint for evaluation result.
    :return best result after adding new result
    """
    return {m: new_result for m in params.test.metrics}
    ret = load_results(mode, params)
    ret["evaluations"].append(new_result)
    for m in params.test.metrics:
        if m not in ret["best_results"] or \
                is_better_result(m, ret['best_results'][m]['result'][m], new_result['result'][m]):
            ret["best_results"][m] = new_result
    with open(os.path.join(params.log_dir, "results_%s.json" % mode), "w") as f:
        f.write(json.dumps(ret, indent=2))
    return ret["best_results"]


def log_outputs(mode, params, outputs):
    os.makedirs(params.log_dir, exist_ok=True)
    with open(os.path.join(params.log_dir, "outputs_%s.json" % mode), "w") as f:
        f.write(json.dumps(outputs, indent=2))


def json_dumps(obj):
    return json.dumps(obj, indent=2)
