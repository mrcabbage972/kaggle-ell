import logging
import os.path
from abc import ABC, abstractmethod
from typing import Mapping

import pandas as pd

from competition_data_manager import CompetitionDataManager
from utils import write_git_hash_to_file

logger = logging.getLogger(__name__)


class Solution(ABC):
    def __init__(self, solution_cfg: Mapping, env_cfg: Mapping):
        self.env_cfg = env_cfg
        self.solution_cfg = solution_cfg
        self.competition_data_manager = CompetitionDataManager(self.env_cfg.competition_data_path)

    def train(self):
        logger.info('Starting training')
        train_data = self.competition_data_manager.load_train_data()

        if not os.path.exists(self.env_cfg.artifacts_path):
            os.makedirs(self.env_cfg.artifacts_path)

        write_git_hash_to_file(os.path.join(self.env_cfg.artifacts_path, 'git_hash.txt'))

        self.do_train(train_data, self.solution_cfg.train, self.solution_cfg.model, self.env_cfg.artifacts_path)
        logger.info('Finished training')

    def predict(self) -> pd.DataFrame:
        logger.info('Starting inference')
        test_data = self.competition_data_manager.load_test_data()
        preds = self.do_predict(test_data, self.solution_cfg.inference, self.env_cfg.artifacts_path)
        logger.info('Finished inference')
        return preds

    def create_submission(self):
        preds = self.predict()
        self.competition_data_manager.create_submission_file(preds, self.env_cfg.submission_path)

    @abstractmethod
    def do_train(self, train_data: pd.DataFrame, train_cfg: Mapping, model_cfg: Mapping, artifacts_path: str):
        raise NotImplementedError()

    @abstractmethod
    def do_predict(self, input_data: pd.DataFrame, inference_cfg: Mapping, artifacts_path: str):
        raise NotImplementedError()
