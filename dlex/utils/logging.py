import logging
import os

logger = logging.getLogger('dlex')
logger.setLevel(logging.INFO)
logging.basicConfig()

# Here we define our formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')


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
