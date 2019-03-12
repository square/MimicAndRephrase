import logging
import datetime
from contextlib import contextmanager
import traceback
import sys


# Replace the except hook to make sure exceptions are logged in log files
def logging_except_hook(exception_type, value,  traceback):
    if __logger is not None:
        error("Uncaught exception. ", value)
    sys.__excepthook__(exception_type, value, traceback)


__logger = None
__indent = 0
__start_times = []


def init(logfile=None, name: str=None, override_excepthook: bool=True):
    """

    :param logfile:
    :param name:
    :param override_excepthook: overrides excepthook to make sure exceptions are logged.
    This interferes with intellij's debugger, so you may want to turn it of when debugging.
    :return:
    """
    # replace exception hook only if logfile was initialized
    if override_excepthook:
        sys.excepthook = logging_except_hook
    global __logger
    if name is None:
        name = "default"
    __logger = logging.getLogger(name)
    if logfile:
        file = logging.FileHandler(logfile)
        file.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s]:  %(message)s'))
        __logger.addHandler(file)
    __logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s]:  %(message)s'))
    __logger.addHandler(console)


def color(text: str, color_desc: str) -> str:
    from colorama import Style
    return color_desc + text + Style.RESET_ALL


def fmt(msg, prefix=''):
    return prefix + ('  ' * __indent) + str(msg)


def debug(msg):
    if __logger:
        __logger.debug(fmt(msg))


def log(msg):
    if __logger:
        __logger.info(fmt(msg, '   '))
    else:
        print(fmt(msg), file=sys.stderr)


def info(msg):
    log(msg)


def warn(msg, exception: Exception=None):
    if exception is not None:
        msg = msg + "\n" + __format_exception(exception)
    if __logger:
        __logger.warning(fmt(msg, ''))
    else:
        print(fmt(msg), file=sys.stderr)


def error(msg, exception: Exception=None):
    if exception is not None:
        msg = msg + "\n" + __format_exception(exception)
    if __logger:
        __logger.error(fmt(msg, '  '))
    else:
        print(fmt(msg), file=sys.stderr)


def __format_exception(ex: Exception) -> str:
    return "".join(traceback.format_tb(ex.__traceback__)) + "\n" + str(type(ex)) + ": " + str(ex)


def start_track(track_name):
    global __indent
    global __start_times
    if __logger:
        __logger.info(fmt(track_name, '   ') + " {")
    else:
        print(fmt(track_name) + " {", file=sys.stderr)
    __indent += 1
    __start_times.append(datetime.datetime.now())


def end_track():
    global __indent
    global __start_times
    start_time = __start_times.pop()
    if __indent > 0:
        __indent -= 1
    if __logger:
        __logger.info(fmt("} ", '   ') + str(datetime.datetime.now() - start_time))
    else:
        print(fmt("} ") + str(datetime.datetime.now() - start_time), file=sys.stderr)


@contextmanager
def track(track_name):
    start_track(track_name)
    yield
    end_track()


LABEL = {
    True: "✓",
    False: "✗"
}
