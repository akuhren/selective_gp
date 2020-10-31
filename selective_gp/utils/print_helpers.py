#!/usr/bin/env python

import sys


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def bold(s):
    return f"{bcolors.BOLD}{s}{bcolors.ENDC}"


def green(s):
    return f"{bcolors.OKGREEN}{s}{bcolors.ENDC}"
