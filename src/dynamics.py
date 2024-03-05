#!/usr/bin/env python3
"""Molecular dynamics."""
import jax
import jax.numpy as jnp
from jax import random
from functools import partial


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


def compute_forces(pos, box_size, epsilon=1.0, sigma=1.0):
    """Compute the forces."""
    num_particule = pos.shape[0]
    # distance matrix in 3D
    rij = pos[:, None, :] - pos[None, :, :]
    # PBC
    rij = rij - jnp.round(rij / box_size) * box_size
    # Distance matrix
    r = jnp.sqrt(jnp.sum(rij ** 2, axis=-1))
    r = jnp.where(r < 1e-6, 1e-6, r)
    # Unit vector
    uij = rij / r[:, :, None]
    # Compute the forces from the grad
    lennard_jones_fixed = partial(lennard_jones, epsilon=epsilon, sigma=sigma)
    grad = jax.grad(lennard_jones_fixed)
    f = - jax.vmap(grad)(r.reshape(-1))
    f = f.at[::num_particule + 1].set(0)
    f = f.reshape((num_particule, num_particule))[:, :, None] * uij
    f = jnp.sum(f, axis=0)
    return f


if __name__ == "__main__":
    num_particule = 10
    box_size = 10.0
    key = random.PRNGKey(0)
    pos, vel = initialize_system(num_particule, box_size, key)
    f = compute_forces(pos, box_size)
    print(f)