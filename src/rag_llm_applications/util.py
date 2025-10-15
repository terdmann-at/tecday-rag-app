import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Union

import yaml

logger = logging.getLogger("rag_llm_applications")

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load the config from the specified yaml file

    :param config_file: path of the config file to load
    :return: the parsed config as dictionary
    """
    with open(config_file, "r") as fp:
        config = yaml.safe_load(fp)

    for data_path in config["paths"]["data"]:
        config["paths"]["data"][data_path] = str(Path(config["paths"]["data"][data_path]))

    config["qdrant"]["storage_path"] = str(Path(config["qdrant"]["storage_path"]))

    return config


def logging_setup(config: Dict):
    """
    setup logging based on the configuration

    :param config: the parsed config tree
    """
    log_conf = config["logging"]
    fmt = log_conf["format"]
    if log_conf["enabled"]:
        level = logging._nameToLevel[log_conf["level"].upper()]
    else:
        level = logging.NOTSET
    logging.basicConfig(format=fmt, level=logging.WARNING)
    logger.setLevel(level)


def delete_all_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

