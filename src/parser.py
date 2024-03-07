#!/usr/bin/env python3
"""Parsing the input file for the dynamic."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import yaml


def get_input_file():
    parser = argparse.ArgumentParser(
        description='Parser for input file for the dynamic.'
    )
    parser.add_argument(
        'param_file',
        type=Path,
        help='This is the input file for the dynamic'
    )
    return parser.parse_args().param_file


def get_argument(file):
    """Get the argument from the yaml file."""
    with open(file) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def read_xyz_atomic(file):
    """Read the xyz file."""
    with open(file) as f:
        lines = f.readlines()
        num_particule = int(lines[0])
        comment = lines[1]
        pos = jnp.array(
            [
                [float(i) for i in line.split()[1:]]
                for line in lines[2:]
            ]
        )
    return num_particule, comment, pos


if __name__ == "__main__":
    file = 'system.xyz'
    num_particule, comment, pos = read_xyz_atomic(file)
    # file = get_input_file()
    # argument = get_argument(file.param_file)
    # key = argument['key']
    # print(key)
