import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Solution(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, input_data_path: str, artifacts_pat: str):
        pass

    @abstractmethod
    def predict(self, input_data_path: str, artifacts_path: str):
        pass

    @abstractmethod
    def create_submission(self, test_preds, output_dir: str):
        pass
