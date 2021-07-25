import logging
import os

import toml

from utility_functions import ensure_file, ensure_dir

CONFIG_FILE = './config.toml'


class Config:
    config_file = ensure_file(CONFIG_FILE)
    config = toml.load(config_file)
    parameters = config.get('parameters')
    logging_level = parameters.get('logging_level', 'DEBUG')
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S',
                        level=logging_level)
    logger = logging.getLogger(__name__)
    env = config.get('environment')
    env_paths = env.get('paths')
    _root_dir = env_paths.get('root_dir')
    if _root_dir is None:
        _root_dir = os.getcwd()
    _root_dir = ensure_dir(_root_dir, False)
    QREL_FILE = os.path.join(_root_dir, env_paths.get('qrel_file'))
    RESULTS_DIR = ensure_dir(os.path.join(_root_dir, env_paths.get('results_dir')), True)

    @staticmethod
    def get_logger():
        return Config.logger
