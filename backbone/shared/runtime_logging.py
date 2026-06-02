import datetime
import logging
import logging.handlers
import os
import sys

import requests

from config.backbone.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None

# ===== Custom log levels for this project =====
# Keep standard DEBUG=10. Define INFER/TRAIN as distinct levels above INFO.
INFER_LEVEL = 21
TRAIN_LEVEL = 22
logging.addLevelName(INFER_LEVEL, "INFER")
logging.addLevelName(TRAIN_LEVEL, "TRAIN")


def _env_log_level(default: str = "INFER") -> str:
    raw = os.environ.get("PYPRISM_LOG_LEVEL") or os.environ.get("PYMCIT_LOG_LEVEL", default)
    return str(raw).strip().upper()


def set_log_level(level: str) -> None:
    """
    Set global log level.

    Supported: DEBUG / INFER / TRAIN / INFO / WARNING / ERROR / CRITICAL
    """
    lvl = str(level).strip().upper()
    root = logging.getLogger()
    if lvl == "DEBUG":
        root.setLevel(logging.DEBUG)
    elif lvl == "INFER":
        root.setLevel(INFER_LEVEL)
    elif lvl == "TRAIN":
        root.setLevel(TRAIN_LEVEL)
    else:
        root.setLevel(getattr(logging, lvl, logging.INFO))


def configure_prism_logging_from_env(default_when_unset: str = "TRAIN") -> str:
    """
    Normalize ``PYPRISM_LOG_LEVEL`` on ``os.environ`` and apply it to the stdlib root logger.

    Training / HF often install handlers before this runs (e.g. at INFO). For ``DEBUG`` we
    lower existing handlers and strip ``_ProjectLevelFilter`` so ``logging.getLogger(...).debug``
    (e.g. custom ``method.custom.*.integration``) actually reaches stderr / tee logs.

    ``PYMCIT_LOG_LEVEL`` is accepted as a legacy alias.
    """
    raw = os.environ.get("PYPRISM_LOG_LEVEL") or os.environ.get(
        "PYMCIT_LOG_LEVEL", default_when_unset
    )
    level = str(raw).strip().upper()
    os.environ["PYPRISM_LOG_LEVEL"] = level
    root = logging.getLogger()
    set_log_level(level)
    if level != "DEBUG":
        return level
    fmt = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    if not root.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setLevel(logging.DEBUG)
        h.setFormatter(fmt)
        root.addHandler(h)
    else:
        for h in root.handlers:
            h.setLevel(logging.DEBUG)
            if h.formatter is None:
                h.setFormatter(fmt)
            for f in list(getattr(h, "filters", []) or []):
                if isinstance(f, _ProjectLevelFilter):
                    h.removeFilter(f)
    return level


def configure_pymcit_logging_from_env(default_when_unset: str = "TRAIN") -> str:
    """Legacy alias for :func:`configure_prism_logging_from_env`."""
    return configure_prism_logging_from_env(default_when_unset)


class _ProjectLevelFilter(logging.Filter):
    """
    Filter that maps our custom levels onto a simple threshold.
    """

    _order = {
        "DEBUG": logging.DEBUG,
        "INFER": INFER_LEVEL,
        "TRAIN": TRAIN_LEVEL,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    def __init__(self, level_name: str):
        super().__init__()
        self.threshold = self._order.get(level_name.upper(), INFER_LEVEL)

    def filter(self, record: logging.LogRecord) -> bool:
        return int(record.levelno) >= int(self.threshold)


def get_logger(name: str) -> logging.Logger:
    """
    Lightweight logger accessor. Respects env ``PYPRISM_LOG_LEVEL`` (legacy: ``PYMCIT_LOG_LEVEL``).
    """
    logger = logging.getLogger(name)
    # Ensure root has at least one handler/formatter
    if not logging.getLogger().handlers:
        logging.basicConfig(level=INFER_LEVEL)
    # Apply level and filter once
    level_name = _env_log_level()
    set_log_level(level_name)
    for h in logging.getLogger().handlers:
        if not any(isinstance(f, _ProjectLevelFilter) for f in getattr(h, "filters", [])):
            h.addFilter(_ProjectLevelFilter(level_name))
    return logger


def log_infer(msg: str, *, logger: logging.Logger | None = None) -> None:
    (logger or get_logger("infer")).log(INFER_LEVEL, msg)


def log_train(msg: str, *, logger: logging.Logger | None = None) -> None:
    (logger or get_logger("train")).log(TRAIN_LEVEL, msg)


def log_debug(msg: str, *, logger: logging.Logger | None = None) -> None:
    (logger or get_logger("debug")).debug(msg)


def is_debug() -> bool:
    return _env_log_level().upper() == "DEBUG"


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"