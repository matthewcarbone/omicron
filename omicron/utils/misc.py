#!/usr/bin/env python

__author__ = "Matthew Carbone"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"


import datetime


def current_datetime():
    """Get's the current date and time and converts it into a string to use
    for tagging files."""

    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S")
