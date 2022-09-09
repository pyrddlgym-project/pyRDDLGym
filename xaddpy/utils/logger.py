import logging
import os
from os import path


def set_log_file_path(args):
    dir_path = path.join(args.results_dir, 'logs')
    os.makedirs(dir_path, exist_ok=True)
    fname = f'{args.model_name}_{args.date_time}.log'

    # log will not be overwritten due to the time stamp
    if '.log' not in fname:
        fname = fname + '.log'

    # Remove previously added file handlers
    for hdlr in logger.handlers[:]:
        if isinstance(hdlr, logging.FileHandler):
            logger.removeHandler(hdlr)

    # Create and add a new handler
    file_handler = logging.FileHandler(path.join(dir_path, fname), 'w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return dir_path


def setup_custom_logger():
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger


logger = setup_custom_logger()