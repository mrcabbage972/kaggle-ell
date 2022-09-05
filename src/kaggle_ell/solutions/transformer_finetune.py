from typing import Mapping

import pandas as pd

from kaggle_ell.solution_factory import SolutionFactory
from kaggle_ell.solution import Solution


@SolutionFactory.register('transformer_finetune')
class TransformerFinetune(Solution):
    def do_train(self, train_data: pd.DataFrame, train_cfg: Mapping, model_cfg: Mapping, artifacts_path: str):
        pass

    def do_predict(self, input_data: pd.DataFrame, inference_cfg: Mapping, artifacts_path: str):
        pass