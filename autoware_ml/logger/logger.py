import datetime
import logging
from logging import FileHandler, StreamHandler, getLogger
import os
import uuid


class Configurations(object):
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file_path = os.getenv(
        "LOG_FILE_PATH", f"/tmp/log/{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
    )

def CustomTextFormatter():
    return logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d %(funcName)s] %(message)s"
    )


class SensitiveWordFilter(logging.Filter):
    def filter(self, record):
        sensitive_words = [
            "password",
            "auth_token",
            "token",
            "ingest.sentry.io",
            "secret",
        ]
        log_message = record.getMessage()
        for word in sensitive_words:
            if word in log_message:
                return False
        return True


def configure_logger(
    log_file_path: str = Configurations.log_file_path,
    modname: str = __name__,
):
    log_directory = os.path.dirname(log_file_path)
    os.makedirs(log_directory, exist_ok=True)

    logger = getLogger(modname)
    logger.addFilter(SensitiveWordFilter())
    logger.setLevel(Configurations.log_level)

    formatter = CustomTextFormatter()

    sh = StreamHandler()
    sh.setLevel(Configurations.log_level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = FileHandler(log_file_path)
    fh.setLevel(Configurations.log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def log_decorator(logger=configure_logger()):
    def _log_decorator(func):
        def wrapper(*args, **kwargs):
            job_id = str(uuid.uuid4())[:8]
            logger.debug(f"START {job_id} func:{func.__name__} args:{args} kwargs:{kwargs}")
            res = func(*args, **kwargs)
            logger.debug(f"RETURN FROM {job_id} return:{res}")
            return res

        return wrapper

    return _log_decorator
