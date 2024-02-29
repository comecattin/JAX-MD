#!/usr/bin/env python3
"""Parsing the input file for the dynamic."""

import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(
        description='Parser for input file for the dynamic.'
    )
    parser.add_argument(
        'param_file',
        type=Path,
        help='This is the input file for the dynamic'
    )
    return parser.parse_args()

if __name__ == "__main__":
    pass
