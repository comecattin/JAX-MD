#!/usr/bin/env python3
"""Parsing the input file for the dynamic."""

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import jax.numpy as jnp
import yaml
from jax import random

from .config import SimulationConfig, finite_number, positive_integer, validate_dynamics


class Parser:
    """Parser for the input file for the dynamic."""

    def __init__(self):
        self.file = self.get_input_file()
        self.arguments = self.get_argument()
        self.xyz_file = self.arguments['xyz_file']
        self.atom_type, self.pos, self.vel = self.get_position_velocities()
        self.display_energy = self.arguments['display_energy']
        self.display_animation = self.arguments['display_animation']
        validate_dynamics(**self.get_dynamics_kwargs())

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
        try:
            with open(self.file) as stream:
                values = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f'Invalid YAML in {self.file}: {exc}') from exc
        self.config = SimulationConfig.from_mapping(values, self.file.parent)
        return asdict(self.config)

    def get_position_velocities(self):
        """Load XYZ coordinates or generate the configured initial state."""
        if self.xyz_file is not None:
            _, atom_type, pos, vel = self.read_xyz_atomic()
            return atom_type, pos, vel
        pos, vel = self.initialize_system(
            self.arguments['num_particule'], self.arguments['box_size'],
            key=random.PRNGKey(self.arguments['key']),
            initialization=self.arguments['initialization'],
            temperature=self.arguments['temperature'],
        )
        return ['X'] * pos.shape[0], pos, vel

    @staticmethod
    def initialize_system(num_particule, box_size, key,
                          initialization='lattice', temperature=1.0):
        """Initialize unit-mass particles in reduced units (Boltzmann k=1).

        Lattice positions avoid accidental close pairs. Gaussian velocities
        have zero center-of-mass momentum and are scaled to the requested
        temperature using 3N-3 degrees of freedom.
        """
        positive_integer('num_particule', num_particule)
        finite_number('box_size', box_size)
        finite_number('temperature', temperature, allow_zero=True)
        if num_particule == 1 and temperature > 0:
            raise ValueError('Positive temperature requires at least two particles '
                             'when center-of-mass momentum is removed.')
        key, subkey_pos, subkey_vel = random.split(key, 3)
        if initialization == 'lattice':
            side = math.ceil(num_particule ** (1 / 3))
            grid = jnp.stack(jnp.meshgrid(
                *([jnp.arange(side)] * 3), indexing='ij'
            ), axis=-1).reshape(-1, 3)
            pos = (grid[:num_particule] + 0.5) * (box_size / side)
        elif initialization == 'random':
            pos = random.uniform(subkey_pos, (num_particule, 3)) * box_size
        else:
            raise ValueError('initialization must be lattice or random.')
        if temperature == 0:
            return pos, jnp.zeros_like(pos)
        vel = random.normal(subkey_vel, (num_particule, 3))
        vel = vel - jnp.mean(vel, axis=0)
        vel = vel * jnp.sqrt(
            (3 * num_particule - 3) * temperature / jnp.sum(vel ** 2)
        )

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
            if not lines:
                raise ValueError('XYZ file is empty.')
            try:
                num_particule = int(lines[0])
            except ValueError as exc:
                raise ValueError('XYZ atom count must be a positive integer.') from exc
            positive_integer('XYZ atom count', num_particule)
            # A coordinate-shaped first line selects the legacy comment-free
            # format; otherwise the first line after the count is a comment.
            coordinates = lines[1:]
            first = coordinates[0].split() if coordinates else []
            first_is_atom = len(first) == 4
            if first_is_atom:
                try:
                    [float(value) for value in first[1:]]
                except ValueError:
                    first_is_atom = False
            if not first_is_atom:
                coordinates = coordinates[1:]
            while coordinates and not coordinates[-1].strip():
                coordinates.pop()
            if len(coordinates) != num_particule:
                raise ValueError('XYZ atom count does not match coordinates.')
            rows = [line.split() for line in coordinates]
            if any(len(row) != 4 for row in rows):
                raise ValueError('XYZ atoms must have three coordinates.')
            try:
                values = [[float(value) for value in row[1:]] for row in rows]
            except ValueError as exc:
                raise ValueError('XYZ coordinates must be numeric.') from exc
            limit = float(jnp.finfo(jnp.asarray(0.).dtype).max)
            if any(not math.isfinite(value) or abs(value) > limit
                   for row in values for value in row):
                raise ValueError('XYZ coordinates must be finite in simulation precision.')
            pos = jnp.array(values)
            atom_type = [line.split()[0] for line in coordinates]
        vel = jnp.zeros_like(pos)
        return num_particule, atom_type, pos, vel

    def get_dynamics_kwargs(self):
        """Get the keyword arguments for the dynamics.

        Returns
        -------
        dict: Dictionary of the keyword arguments for the dynamics.
        """
        try:
            center = self.arguments['center']
        except KeyError:
            center = False
        return {
            'box_size': self.arguments['box_size'],
            'position': self.pos,
            'velocity': self.vel,
            'dt': self.arguments['dt'],
            'n_steps': self.arguments['n_steps'],
            'epsilon': self.arguments['epsilon'],
            'sigma': self.arguments['sigma'],
            'printing_step': self.arguments['printing_step'],
            'writing_step': self.arguments['writing_step'],
            'center': center,
        }


if __name__ == "__main__":

    parser = Parser()
