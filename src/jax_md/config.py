"""Validated simulation configuration and public API input checks."""
import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def positive_integer(name, value):
    """Require a positive integer, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer.')


def finite_number(name, value, *, allow_zero=False):
    """Require a finite positive (or nonnegative) real number."""
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value)
            or (value < 0 if allow_zero else value <= 0)):
        bound = 'nonnegative' if allow_zero else 'positive'
        raise ValueError(f'{name} must be {bound} and finite.')


def validate_dynamics(position, velocity, **parameters):
    """Validate once at the host boundary before entering compiled dynamics."""
    for name in ('box_size', 'dt', 'sigma'):
        finite_number(name, parameters[name])
    finite_number('epsilon', parameters['epsilon'], allow_zero=True)
    for name in ('n_steps', 'printing_step', 'writing_step'):
        positive_integer(name, parameters[name])
    if not isinstance(parameters['center'], bool):
        raise ValueError('center must be a boolean.')
    for name, array in (('position', position), ('velocity', velocity)):
        try:
            array = jnp.asarray(array)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must contain real numeric coordinates.') from exc
        if not (jnp.issubdtype(array.dtype, jnp.floating)
                or jnp.issubdtype(array.dtype, jnp.integer)):
            raise ValueError(f'{name} must contain real numeric coordinates.')
        if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != 3:
            raise ValueError(f'{name} must have shape (N, 3) with N >= 1.')
        if not bool(jnp.all(jnp.isfinite(array))):
            raise ValueError(f'{name} must contain only finite coordinates.')
    if jnp.shape(position) != jnp.shape(velocity):
        raise ValueError('position and velocity must have matching shapes.')
    # Check minimum-image coincidences on the host before invoking kernels.
    # Iterate over particles to avoid allocating another dense pair matrix.
    coordinates = np.asarray(jnp.asarray(position), dtype=float)
    box_size = parameters['box_size']
    for i in range(len(coordinates) - 1):
        displacement = coordinates[i + 1:] - coordinates[i]
        displacement -= np.round(displacement / box_size) * box_size
        overlaps = np.flatnonzero(np.all(displacement == 0, axis=1))
        if overlaps.size:
            j = i + 1 + int(overlaps[0])
            raise ValueError(
                f'Initial particles {i} and {j} overlap under periodic '
                f'boundaries (box_size={box_size}).'
            )


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration defaults shared by YAML parsing and the CLI."""

    box_size: float
    dt: float = 0.001
    n_steps: int = 1000
    epsilon: float = 1.0
    sigma: float = 1.0
    printing_step: int = 100
    writing_step: int = 100
    center: bool = False
    display_energy: bool = False
    display_animation: bool = False
    xyz_file: object = None
    key: object = None
    num_particule: object = None
    initialization: str = 'lattice'
    temperature: float = 1.0

    @classmethod
    def from_mapping(cls, values, directory):
        """Validate a YAML mapping and resolve XYZ paths relative to it."""
        if not isinstance(values, dict) or not values:
            raise ValueError('Configuration must be a nonempty YAML mapping.')
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError('Unknown configuration fields: '
                             + ', '.join(sorted(map(str, unknown))))
        if 'box_size' not in values:
            raise ValueError('Missing required field: box_size.')
        config = cls(**values)
        for name in ('box_size', 'dt', 'sigma'):
            finite_number(name, getattr(config, name))
        for name in ('epsilon', 'temperature'):
            finite_number(name, getattr(config, name), allow_zero=True)
        for name in ('n_steps', 'printing_step', 'writing_step'):
            positive_integer(name, getattr(config, name))
        for name in ('center', 'display_energy', 'display_animation'):
            if not isinstance(getattr(config, name), bool):
                raise ValueError(f'{name} must be a boolean.')
        if config.initialization not in ('lattice', 'random'):
            raise ValueError('initialization must be lattice or random.')
        if config.key is not None:
            if (isinstance(config.key, bool) or not isinstance(config.key, int)
                    or not 0 <= config.key <= 2**32 - 1):
                raise ValueError('key must be an integer between 0 and 4294967295.')
        if config.num_particule is not None:
            positive_integer('num_particule', config.num_particule)
        if config.xyz_file is None:
            for name in ('key', 'num_particule'):
                if getattr(config, name) is None:
                    raise ValueError(f'Missing required field: {name} (no xyz_file).')
            if config.num_particule == 1 and config.temperature > 0:
                raise ValueError('Positive temperature requires at least two particles.')
        else:
            if not isinstance(config.xyz_file, str) or not config.xyz_file.strip():
                raise ValueError('xyz_file must be a nonempty path string or null.')
            values = asdict(config)
            values['xyz_file'] = str((Path(directory) / config.xyz_file).resolve())
            config = cls(**values)
        return config
