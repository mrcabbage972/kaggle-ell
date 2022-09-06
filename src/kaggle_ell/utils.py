import collections
import logging
import shutil

import git
import tqdm

logger = logging.getLogger(__name__)


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def register_tqdm_logger():
    log = logging.getLogger()
    log.addHandler(TqdmLoggingHandler())


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_disk_usage():
    BytesPerGB = 1024 * 1024 * 1024

    (total, used, free) = shutil.disk_usage(".")
    logger.info(
        "Disk Usage - total: %.2fGB" % (float(total) / BytesPerGB) + ", Used:  %.2fGB" % (float(used) / BytesPerGB))

def get_git_hash():
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return sha
    except:
        logger.warning('Failed getting Git hash')
        return 'none'


def write_git_hash_to_file(output_file_path: str):
    with open(output_file_path, 'w') as fout:
        fout.write(get_git_hash())