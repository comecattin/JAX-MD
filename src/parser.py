#!/usr/bin/env python3
"""Parsing the input file for the dynamic."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import yaml


class Parser:
    """Parser for the input file for the dynamic."""

    def __init__(self):
        self.file = self.get_input_file()
        self.arguments = self.get_argument()

    def get_input_file(self):
        """Get the input file for the dynamic.

        The input file is a yaml file.
        """
        parser = argparse.ArgumentParser(
            description='Parser for input file for the dynamic.'
        )
        parser.add_argument(
            'param_file',
            type=Path,
            help='This is the input file for the dynamic'
        )
        return parser.parse_args().param_file

    def get_argument(self):
        """Get the argument from the input file.

        Returns
        -------
            dict: Dictionary of the arguments.
        """
        with open(self.file) as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)


def read_xyz_atomic(file):
    """Extract system information from a xyz file.

    Parameters
    ----------
    file : str
        Path to the xyz file.

    Returns
    -------
    num_particule : int
        Number of particules in the system.
    comment : str
        Comment of the system.
    pos : jnp.array
        Position of the particules in the system.
    """
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
    file = 'data/xyz/system.xyz'
    num_particule, comment, pos = read_xyz_atomic(file)
    # file = get_input_file()
    # argument = get_argument(file.param_file)
    # key = argument['key']
    # print(key)
    parser = Parser()
    print(parser.arguments)
