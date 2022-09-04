import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging.config
import os
import pathlib

import wandb
import hydra
import yaml
from omegaconf import OmegaConf

from logging_utils import register_tqdm_logger, flatten, log_disk_usage
from solution_factory import SolutionFactory

HERE = pathlib.Path(__file__).parent.resolve()

logging.config.dictConfig(yaml.safe_load(
    pathlib.Path(os.path.join(HERE, 'config', 'logging.yaml')).read_text()))

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: OmegaConf):
    register_tqdm_logger()
    logger.info('cwd={}'.format(os.getcwd()))
    log_disk_usage()

    wandb_mode = 'online' if cfg.wandb.enabled else 'disabled'

    wandb.init(project=cfg.wandb.project, config=flatten(dict(cfg)), mode=wandb_mode)

    solution = SolutionFactory.make(cfg.solution.args, cfg.env)

    if cfg.solution.do_train:
        solution.train(cfg.env.raw_data_path, cfg.env.artifacts_path)
    else:
        logger.info('Skipping training due to config')

    if cfg.solution.do_predict:
        if os.path.exists(cfg.env.artifacts_path):
            preds = solution.predict(cfg.env.raw_data_path, cfg.env.artifacts_path)
            preds.to_pickle(os.path.join(cfg.env.submission_path, 'preds.pkl'))
            solution.create_submission(preds, cfg.env.submission_path)
        else:
            logger.error('Artifacts dir not found')
    else:
        logger.info('Skipping inference due to config')

    log_disk_usage()
    logger.info('Finished successfully')


if __name__ == '__main__':
    main()

