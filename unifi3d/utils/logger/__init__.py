from .logger import Logger


def get_logger(config) -> Logger:
    return Logger(config)
