import logging

name = "Deep Research"


def setup_logger():
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.propagate = True
    return logger
