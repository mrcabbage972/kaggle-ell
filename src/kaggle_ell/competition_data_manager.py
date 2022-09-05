import logging
import os
from typing import Collection

import pandas as pd

logger = logging.getLogger(__name__)

class CompetitionDataManager:
    LABEL_COLUMNS = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

    def __init__(self, competition_data_path: str):
        self.competition_data_path = competition_data_path

    def load_train_data(self) -> pd.DataFrame:
        logger.info('Loading train data')
        return pd.read_csv(os.path.join(self.competition_data_path, 'train.csv'))

    def load_test_data(self) -> pd.DataFrame:
        logger.info('Loading test data')
        return pd.read_csv(os.path.join(self.competition_data_path, 'test.csv'))

    def load_sample_submission(self) -> pd.DataFrame:
        logger.info('Loading sample submission')
        return pd.read_csv(os.path.join(self.competition_data_path, 'sample_submission.csv'))

    def create_submission_file(self, preds: Collection, output_dir: str):
        logger.info('Creating submission file')
        test_df = self.load_test_data()
        output_df = pd.concat([test_df[['text_id']], pd.DataFrame(preds, columns=self.LABEL_COLUMNS)], axis=1)

        if not os.path.exists(output_dir):
            logger.info(f'Creating output dir {output_dir}')
            os.makedirs(output_dir)

        output_df.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)

