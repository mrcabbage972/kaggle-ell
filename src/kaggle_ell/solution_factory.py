import logging
from collections.abc import Callable
from typing import Mapping

from kaggle_ell.solution import Solution


logger = logging.getLogger(__name__)


class SolutionFactory:
    registry = {}
    """ Internal registry for available solutions """

    @classmethod
    def register(cls, name: str) -> Callable:

        def inner_wrapper(wrapped_class: Solution) -> Callable:
            if name in cls.registry:
                logger.warning('Solution %s already exists. Will replace it', name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def make(cls, name: str, solution_cfg: Mapping, env_cfg: Mapping) -> Solution:
        """ Factory command to create the solution """
        logger.info('Creating solution ' + name)
        solution_class = cls.registry[name]
        solution = solution_class(solution_cfg, env_cfg)
        return solution

from kaggle_ell.solutions.dummy_solution import DummySolution
from kaggle_ell.solutions.transformer_finetune import TransformerFinetune