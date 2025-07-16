import sys
from loguru import logger

def get_logger(debug=True):
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    logger.add(sys.stderr, level=level)
    logger.add("file.log", level=level, rotation="500 MB")
    return logger
