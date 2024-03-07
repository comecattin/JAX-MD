#!/usr/bin/env python3
"""Parsing the input file for the dynamic."""

import argparse
from pathlib import Path

import jax.numpy as jnp
import yaml
from dynamics import initialize_system


class Parser:
    """Parser for the input file for the dynamic."""

    def __init__(self):
        self.file = self.get_input_file()
        self.arguments = self.get_argument()
        self.xyz_file = self.arguments['xyz_file']
        self.pos, self.vel = self.get_position_velocities()

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

    def get_position_velocities(self):
        """Get the position and velocities of the system.

        Returns
        -------
        pos : jnp.array
            Position of the particules in the system.
        vel : jnp.array
            Velocity of the particules in the system.
        """
        if 'xyz_file' not in self.arguments:
            try:
                print('No XYZ file provided, initializing the system.')
                pos, vel = initialize_system(
                    self.arguments['num_particule'],
                    self.arguments['box_size'],
                    self.arguments['key']
                )
            except KeyError:
                print(
                    'The argument key or box_size or num_particules is missing.'
                )
        else:
            try:
                print('Reading XYZ file, assuming initial velocity is 0.')
                _, _, pos, vel = self.read_xyz_atomic()
            except KeyError:
                print('The argument xyz_file is missing.')

        return pos, vel

    def read_xyz_atomic(self):
        """Extract system information from a xyz file.

        Returns
        -------
        num_particule : int
            Number of particules in the system.
        comment : str
            Comment of the system.
        pos : jnp.array
            Position of the particules in the system.
        vel : jnp.array
            Velocity of the particules in the system.
        """
        with open(self.xyz_file) as f:
            lines = f.readlines()
            num_particule = int(lines[0])
            comment = lines[1]
            pos = jnp.array(
                [
                    [float(i) for i in line.split()[1:]]
                    for line in lines[2:]
                ]
            )
        vel = jnp.zeros_like(pos)
        return num_particule, comment, pos, vel


if __name__ == "__main__":
    # file = 'data/xyz/system.xyz'
    # num_particule, comment, pos = read_xyz_atomic(file)
    # file = get_input_file()
    # argument = get_argument(file.param_file)
    # key = argument['key']
    # print(key)
    parser = Parser()
    print(parser.arguments)
    print(parser.pos)
    print(parser.vel)
