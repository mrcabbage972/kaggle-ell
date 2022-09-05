from kaggle_ell.solution_factory import SolutionFactory
from kaggle_ell.solution import Solution


@SolutionFactory.register('transformer_finetune')
class TransformerFinetune(Solution):
    def train(self, input_data_path: str, artifacts_pat: str):
        pass

    def predict(self, input_data_path: str, artifacts_path: str):
        pass

    def create_submission(self, test_preds, output_dir: str):
        pass