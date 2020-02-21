import os
import pathlib

import pytest


@pytest.fixture(scope="session", autouse=True)
def init(tmpdir_factory):
    os.chdir(pathlib.Path(__file__).parent.absolute())
    os.environ["DLEX_TMP_PATH"] = str(tmpdir_factory.mktemp('tmp'))
    os.environ["DLEX_DATASETS_PATH"] = str(tmpdir_factory.mktemp('datasets'))
    os.environ["DLEX_SAVED_MODELS_DIR"] = str(tmpdir_factory.mktemp('saved_models'))
    os.environ["DLEX_LOG_DIR"] = str(tmpdir_factory.mktemp('logs'))