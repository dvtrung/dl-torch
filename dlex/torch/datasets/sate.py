import os

from .base import BaseDataset, default_params
from utils.utils import maybe_download, maybe_unzip


class SATE_IV(BaseDataset):
    working_dir = os.path.join("datasets", "sate_iv")
    raw_data_dir = os.path.join(working_dir, "raw")

    def __init__(self, mode, params):
        super().__init__(mode, params)

    @classmethod
    def maybe_download_and_extract(cls, force=False):
        url = "https://samate.nist.gov/SATE4/resources/SATE4data.tar.lzma"
        maybe_download("SATE4data.tar.lzma", cls.working_dir, url)
        maybe_unzip(os.path.join(cls.working_dir, "SATE4data.tar.lzma"), cls.working_dir, "raw")

    @classmethod
    def maybe_preprocess(cls, force=False):

