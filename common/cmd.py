from __future__ import print_function
import sys
import json
from dolfin import MPI


RED = "\033[1;37;31m{s}\033[0m"
BLUE = "\033[1;37;34m{s}\033[0m"
GREEN = "\033[1;37;32m{s}\033[0m"
YELLOW = "\033[1;37;33m{s}\033[0m"
CYAN = "\033[1;37;36m{s}\033[0m"
NORMAL = "{s}"
ON_RED = "\033[41m{s}\033[0m"


__all__ = ["mpi_comm", "mpi_barrier", "mpi_rank", "mpi_size", "mpi_is_root",
           "convert", "str2list", "parseval", "parse_command_line",
           "info_style", "info_red", "info_blue", "info_yellow",
           "info_green", "info_cyan", "info", "info_on_red",
           "info_split_style", "info_split", "info_warning",
           "info_error"]


def mpi_comm():
    return MPI.comm_world


def mpi_barrier():
    MPI.barrier(mpi_comm())


def mpi_rank():
    return MPI.rank(mpi_comm())


def mpi_size():
    return MPI.size(mpi_comm())


def mpi_is_root():
    return mpi_rank() == 0


# Stolen from Oasis
def convert(data):
    if isinstance(data, dict):
        return {convert(key): convert(value)
                for key, value in data.iteritems()}
    elif isinstance(data, list):
        return [convert(element) for element in data]
    else:
        return data


def str2list(string):
    if bool(string[0] == "[" and string[-1] == "]" and
            "--" not in string):
        # Avoid parsing line specification as list.
        li = string[1:-1].split(",")
        for i in range(len(li)):
            li[i] = str2list(li[i])
        return li
    else:
        return parseval(string)


def parseval(value):
    try:
        value = json.loads(value)
    except ValueError:
        # json understands true/false, not True/False
        if value in ["True", "False"]:
            value = eval(value)
        elif "True" in value or "False" in value:
            value = eval(value)

    if isinstance(value, dict):
        value = convert(value)
    elif isinstance(value, list):
        value = convert(value)
    return value


def parse_command_line():
    cmd_kwargs = dict()
    for s in sys.argv[1:]:
        if s in ["-h", "--help", "help"]:
            key, value = "help", "true"
        elif s.count('=') == 0:
            key, value = s, "true"
        elif s.count('=') == 1:
            key, value = s.split('=', 1)
        else:
            raise TypeError(
                "Only kwargs separated with at the most a single '=' allowed.")

        value = parseval(value)
        if isinstance(value, str):
            value = str2list(value)

        cmd_kwargs[key] = value
    return cmd_kwargs


def info_style(message, check=True, style=NORMAL):
    if mpi_is_root() and check:
        print(style.format(s=message))


def info_red(message, check=True):
    info_style(message, check, RED)


def info_blue(message, check=True):
    info_style(message, check, BLUE)


def info_yellow(message, check=True):
    info_style(message, check, YELLOW)


def info_green(message, check=True):
    info_style(message, check, GREEN)


def info_cyan(message, check=True):
    info_style(message, check, CYAN)


def info(message, check=True):
    info_style(message, check)


def info_on_red(message, check=True):
    info_style(message, check, ON_RED)


def info_split_style(msg_1, msg_2, style_1=BLUE, style_2=NORMAL, check=True):
    if mpi_is_root() and check:
        print(style_1.format(s=msg_1) + " " + style_2.format(s=msg_2))


def info_split(msg_1, msg_2, check=True):
    info_split_style(msg_1, msg_2, check=check)


def info_warning(message, check=True):
    info_split_style("Warning:", message, style_1=ON_RED, check=check)


def info_error(message, check=True):
    info_split_style("Error:", message, style_1=ON_RED, check=check)
    exit("")
