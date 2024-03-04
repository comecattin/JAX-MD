#!/usr/bin/env python3
"""Molecular dynamics."""
import jax
import jax.numpy as jnp
from jax import random


def initialize_system(num_particule, box_size, key):
    """Initialize the system."""
    key, subkey_pos, subkey_vel = random.split(key, 3)
    # Positions within the box
    pos = random.uniform(subkey_pos, (num_particule, 3)) * box_size
    # Random velocities
    vel = random.normal(subkey_vel, (num_particule, 3))

    return pos, vel


def lennard_jones(r, epsilon=1.0, sigma=1.0):
    """Lennard-Jones potential."""
    r6 = (sigma / r) ** 6
    r12 = r6 ** 2
    return 4.0 * epsilon * (r12 - r6)


if __name__ == "__main__":
    num_particule = 10
    box_size = 10.0
    key = random.PRNGKey(0)
    pos, vel = initialize_system(num_particule, box_size, key)
    print(pos, vel)
