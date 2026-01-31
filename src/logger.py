"""Logging setup for the botnet detection pipeline."""

import logging
import os


def log_and_print(message, log_obj, level='info'):
    """Log a message and print it to the console."""
    if level == 'info':
        log_obj.info(message)
    elif level == 'warning':
        log_obj.warning(message)
    elif level == 'error':
        log_obj.error(message)
    print(message)


def setup_logging(log_folder='logs'):
    """
    Create and configure the main logger with file handlers for
    info, warnings, and errors. Returns the configured logger.
    """
    os.makedirs(log_folder, exist_ok=True)

    logger = logging.getLogger('botnet_detection')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    log_files = {
        'analysis.log': logging.INFO,
        'errors.log': logging.ERROR,
        'warnings.log': logging.WARNING,
    }

    for filename, level in log_files.items():
        path = os.path.join(log_folder, filename)
        if os.path.exists(path):
            os.remove(path)
        handler = logging.FileHandler(path)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
