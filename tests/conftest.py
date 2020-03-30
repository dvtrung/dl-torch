import os
import pathlib

import pytest


@pytest.fixture(scope="session", autouse=True)
def init(tmpdir_factory):
    os.chdir(pathlib.Path(__file__).parent.absolute())
    os.environ["DLEX_TMP_PATH"] = str(tmpdir_factory.mktemp('tmp'))
    os.environ["DLEX_DATASET_PATH"] = str(tmpdir_factory.mktemp('datasets'))
    os.environ["DLEX_CHECKPOINT_PATH"] = str(tmpdir_factory.mktemp('checkpoints'))
    os.environ["DLEX_LOG_DIR"] = str(tmpdir_factory.mktemp('logs'))