import logging
import os

import numpy as np
from colorama import Fore, Style

logger = logging.getLogger('dlex')
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    level=logging.INFO,
    format=Fore.GREEN + '%(levelname)s - %(message)s' + Style.RESET_ALL
)
logging.basicConfig(
    level=logging.DEBUG,
    format=Fore.MAGENTA + '%(levelname)s - %(message)s' + Style.RESET_ALL
)
logging.basicConfig(
    level=logging.ERROR,
    format=Fore.RED + '%(levelname)s - %(message)s' + Style.RESET_ALL
)
logging.basicConfig(
    level=logging.WARNING,
    format=Fore.RED + '%(levelname)s - %(message)s' + Style.RESET_ALL
)

# Here we define our formatter
formatter = logging.Formatter(Fore.BLUE + '%(asctime)s - %(levelname)s - %(message)s' + Style.RESET_ALL)


def set_log_dir(params):
    os.makedirs(params.log_dir, exist_ok=True)

    log_info_handler = logging.FileHandler(
        os.path.join(params.log_dir, "info.log"))
    log_info_handler.setLevel(logging.INFO)
    log_info_handler.setFormatter(formatter)

    log_debug_handler = logging.FileHandler(
        os.path.join(params.log_dir, "debug.log"))
    log_debug_handler.setLevel(logging.DEBUG)
    # logDebugHandler.setFormatter(formatter)

    logger.addHandler(log_info_handler)
    logger.addHandler(log_debug_handler)


def beautify(obj):
    if type(obj) is np.ndarray:
        return "[%s]" % ('\t'.join(["%.4f" % x for x in obj]))