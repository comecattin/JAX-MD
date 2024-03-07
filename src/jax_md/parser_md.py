#!/usr/bin/env python3
"""Parsing the input file for the dynamic."""

import argparse
from pathlib import Path

import jax.numpy as jnp
import yaml
from jax import random


class Parser:
    """Parser for the input file for the dynamic."""

    def __init__(self):
        self.file = self.get_input_file()
        self.arguments = self.get_argument()
        try:
            self.xyz_file = self.arguments['xyz_file']
        except KeyError:
            self.xyz_file = None
        self.atom_type, self.pos, self.vel = self.get_position_velocities()

        try:
            self.display_energy = self.arguments['display_energy']
        except KeyError:
            self.display_energy = False
        try:
            self.display_animation = self.arguments['display_animation']
        except KeyError:
            self.display_animation = False

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
                key = random.PRNGKey(self.arguments['key'])
                pos, vel = self.initialize_system(
                    self.arguments['num_particule'],
                    self.arguments['box_size'],
                    key=key
                )
            except KeyError:
                print(
                    'The argument key or box_size or num_particules is missing.'
                )
        else:
            try:
                print('Reading XYZ file, assuming initial velocity is 0.')
                _, atom_type, pos, vel = self.read_xyz_atomic()
            except KeyError:
                print('The argument xyz_file is missing.')

        return atom_type, pos, vel

    @staticmethod
    def initialize_system(num_particule, box_size, key):
        """Initialize the system."""
        key, subkey_pos, subkey_vel = random.split(key, 3)
        # Positions within the box
        pos = random.uniform(subkey_pos, (num_particule, 3)) * box_size
        # Random velocities
        vel = random.normal(subkey_vel, (num_particule, 3))

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
            pos = jnp.array(
                [
                    [float(i) for i in line.split()[1:]]
                    for line in lines[1:]
                ]
            )
            atom_type = [line.split()[0] for line in lines[1:]]
        vel = jnp.zeros_like(pos)
        return num_particule, atom_type, pos, vel

    def get_dynamics_kwargs(self):
        """Get the keyword arguments for the dynamics.

        Returns
        -------
        dict: Dictionary of the keyword arguments for the dynamics.
        """
        return {
            'box_size': self.arguments['box_size'],
            'position': self.pos,
            'velocity': self.vel,
            'dt': self.arguments['dt'],
            'n_steps': self.arguments['n_steps'],
            'epsilon': self.arguments['epsilon'],
            'sigma': self.arguments['sigma'],
        }


if __name__ == "__main__":

    parser = Parser()
