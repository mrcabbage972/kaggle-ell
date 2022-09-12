import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging.config
import os
import pathlib

import wandb
import hydra
import yaml
from omegaconf import OmegaConf

from kaggle_ell.utils import register_tqdm_logger, flatten, log_disk_usage, get_git_hash, write_git_hash_to_file
from kaggle_ell.solution_factory import SolutionFactory

HERE = pathlib.Path(__file__).parent.resolve()

logging.config.dictConfig(yaml.safe_load(
    pathlib.Path(os.path.join(HERE, 'config', 'logging.yaml')).read_text()))

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: OmegaConf):
    logger.info('*********STARTING**********')
    git_hash = get_git_hash(cfg.git_hash)
    logger.info('Git hash={}'.format(git_hash))
    register_tqdm_logger()
    logger.info('cwd={}'.format(os.getcwd()))
    log_disk_usage()

    wandb_mode = 'online' if cfg.wandb.enabled else 'disabled'

    wandb.init(project=cfg.wandb.project, config=flatten(dict(cfg)), mode=wandb_mode)

    solution = SolutionFactory.make(cfg.solution.name, cfg.solution.args, cfg.env)

    if cfg.solution.do_train:
        solution.train()

        write_git_hash_to_file(cfg.git_hash, os.path.join(cfg.env.artifacts_path, 'git_hash.txt'))
    else:
        logger.info('Skipping do_train due to config')

    if cfg.solution.do_predict:
        if os.path.exists(cfg.env.artifacts_path):
            preds = solution.predict()
            preds_path = os.path.join(cfg.env.artifacts_path, 'preds.pkl')
            preds.to_pickle(preds_path)
            logger.info(f'Wrote predictions to path {preds_path}')
        else:
            logger.error('Artifacts dir not found')
    else:
        logger.info('Skipping do_predict due to config')

    if cfg.solution.do_create_submission:
        if os.path.exists(cfg.env.artifacts_path):
            solution.create_submission()
        else:
            logger.error('Artifacts dir not found')
    else:
        logger.info('Skipping do_create_submission due to config')

    log_disk_usage()
    logger.info('*****Finished successfully*****')


if __name__ == '__main__':
    main()

