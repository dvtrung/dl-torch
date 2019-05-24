"""NLP Dataset"""

import os
import re
import unicodedata

from .base import BaseDataset


class VoiceDataset(BaseDataset):
    def __init__(self, mode, params):
        super().__init__(mode, params)
