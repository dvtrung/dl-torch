import runpy

from dlex.utils import logger
from .configs import Configs


if __name__ == "__main__":
    configs = Configs(mode="train")
    backend = configs.params.backend
    if backend is None:
        logger.error("No backend specified. Please add it in config file.")
    logger.info("Backend: %s", backend)
    if backend == "sklearn":
        runpy.run_module("dlex.sklearn.train", run_name=__name__)