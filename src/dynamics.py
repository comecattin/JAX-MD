#!/usr/bin/env python3
"""Molecular dynamics."""
import jax
import jax.numpy as jnp


def step(xyz, velocities, forces, dt, masses):
    """Perform a single step of molecular dynamics."""
    dt2 = 0.5 * dt
    dt2m = jnp.asarray(dt2 / masses[:, None], dtype=jnp.float32)
    velocities = velocities + forces * dt2m
    xyz = xyz + velocities * dt
    return xyz, velocities


def compute_forces(xyz, cutoff,epsilon):
    """Compute the forces using the Lennard-Jones potential."""
    forces = jnp.zeros_like(xyz)

    # Compute forces between neighboring atoms
    for i in range(len(xyz)):
        for j in range(i+1, len(xyz)):
            dist = jnp.linalg.norm(xyz[i] - xyz[j])
            if dist < cutoff:
                force = (
                    24 * epsilon * (
                        2 * (cutoff / dist)**12 - (cutoff / dist)**6
                        ) / dist
                    )

                forces[i] -= force
                forces[j] += force

    return forces

if __name__ == "__main__":
    xyz_input = jnp.array(
        [
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.]
        ],
        dtype=jnp.float32
    )

    velocities_input = jnp.array(
        [
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
        ],
        dtype=jnp.float32
    )

    atomic_masses = jnp.array([16, 1, 1])
