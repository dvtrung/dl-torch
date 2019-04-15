import logging, os
import logging.handlers as handlers
import time

logger = logging.getLogger('dl_torch')
logger.setLevel(logging.INFO)
logging.basicConfig()

# Here we define our formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

def set_log_dir(params):
    os.makedirs(params.log_dir, exist_ok=True)

    logInfoHandler = logging.FileHandler(
        os.path.join(params.log_dir, "info.log"))
    logInfoHandler.setLevel(logging.INFO)
    logInfoHandler.setFormatter(formatter)

    logDebugHandler = logging.FileHandler(
        os.path.join(params.log_dir, "debug.log"))
    logDebugHandler.setLevel(logging.DEBUG)
    # logDebugHandler.setFormatter(formatter)

    logger.addHandler(logInfoHandler)
    logger.addHandler(logDebugHandler)
