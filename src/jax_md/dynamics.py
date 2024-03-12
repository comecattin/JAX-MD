#!/usr/bin/env python3
"""Molecular dynamics."""
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from jax_md import visual
from jax_md.parser_md import Parser

from . import writer


@jax.jit
def lennard_jones(
        r: jnp.ndarray,
        epsilon: float = 1.0,
        sigma: float = 1.0,
    ) -> jnp.ndarray:
    """Lenard Jones potential.

    Parameters
    ----------
    r : jnp.ndarray
        Distance between the particules.
        The shape of the array is (n, n) where n is the number of particules.
    epsilon : float, optional
        Epsilon parameter for the lennard-jones potential, by default 1.0
    sigma : float, optional
        Sigma parameter for the lennar-jones potential, by default 1.0

    Returns
    -------
    jnp.ndarray
        Potential energy between the particules.
    """
    r6 = (sigma / r) ** 6
    r12 = r6 ** 2
    return 4.0 * epsilon * (r12 - r6)


@jax.jit
def compute_forces_and_potential_energy(
    pos: jnp.ndarray,
    box_size: float,
    epsilon: float = 1.0,
    sigma: float = 1.0
    ) -> Tuple[jnp.ndarray, float]:
    """Compute the forces and the potential energy.

    This compute the forces and the potential energy between the particules
    for the Lennard-Jones potential, using the grad from the jax library.

    Parameters
    ----------
    pos : jnp.ndarray
        Position of the particules.
        The shape of the array is (n, 3) where n is the number of particules.
    box_size : float
        Size of the simulation box.
    epsilon : float, optional
        Epsilon parameter for the Lennard-Jones potential, by default 1.0
    sigma : float, optional
        Sigma parameter for the Lennar-Jones potential, by default 1.0

    Returns
    -------
    f : jnp.ndarray
        Forces between the particules.
        The shape of the array is (n, 3) where n is the number of particules.
    potential_energy : float
        Potential energy of the system.

    """
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
    f = jnp.sum(f, axis=1)

    # Potential energy
    potential_energy = jnp.sum(
        lennard_jones(
            r,
            epsilon=epsilon,
            sigma=sigma
            ).reshape(-1).at[::num_particule + 1].set(0)
    )

    return f, potential_energy


@jax.jit
def step(
        position: jnp.ndarray,
        velocity: jnp.ndarray,
        force: jnp.ndarray,
        dt: float,
        box_size: float,
        epsilon: float = 1.0,
        sigma: float = 1.0
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Updade the position, velocity and force of the system.

    This function update the position, velocity and force of the system using
    the velocity Verlet algorithm.

    Parameters
    ----------
    position : jnp.ndarray
        Position of the particules.
        The shape of the array is (n, 3) where n is the number of particules.
    velocity : jnp.ndarray
        Velocity of the particules.
        The shape of the array is (n, 3) where n is the number of particules.
    force : jnp.ndarray
        Forces between the particules.
        The shape of the array is (n, 3) where n is the number of particules.
    dt : float
        Time step.
    box_size : float
        Size of the simulation box.
    epsilon : float, optional
        Epsilon parameter for the Lennard-Jones potential, by default 1.0
    sigma : float, optional
        Sigma parameter for the Lennar-Jones potential, by default 1.0

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        New position, velocity and force of the system.
    """
    # Update the position
    new_position = position + velocity * dt + 0.5 * force * dt ** 2
    new_position = jnp.mod(new_position, box_size)
    # Compute the new forces
    new_force, _ = compute_forces_and_potential_energy(
        new_position,
        box_size,
        sigma=sigma,
        epsilon=epsilon
    )
    # Update the velocity
    new_velocity = velocity + 0.5 * (force + new_force) * dt
    return new_position, new_velocity, new_force


def dynamics(
        position: jnp.ndarray,
        velocity: jnp.ndarray,
        dt: float,
        box_size: float,
        epsilon: float = 1.0,
        sigma: float = 1.0,
        n_steps: int = 1000,
        writing_step: int = 100,
        printing_step: int = 100,
        center: bool = True,
    ) -> Tuple[jnp.ndarray, list, list, list]:
    """Run the dynamics of the system.

    Parameters
    ----------
    position : jnp.ndarray
        Position of the particules.
        The shape of the array is (n, 3) where n is the number of particules.
    velocity : jnp.ndarray
        Velocity of the particules.
        The shape of the array is (n, 3) where n is the number of particules.
    dt : float
        Time step.
    box_size : float
        Size of the simulation box.
    epsilon : float, optional
        Epsilon parameter for the Lennard-Jones potential, by default 1.0
    sigma : float, optional
        Sigma parameter for the Lennar-Jones potential, by default 1.0
    n_steps : int, optional
        Number of time steps, by default 1000
    writing_step : int, optional
        Write the system position every writing_step's time step.
    printing_step : int, optional
        Print the system energy every printing_step's time step.
    center : bool, optional
        Center the position of the particules in the box, by default True

    Returns
    -------
    position_list : jnp.ndarray
        List of the position of the particules at each time step.
    kinetic_energy_list : list
        List of the kinetic energy of the system at each time step.
    potential_energy_list : list
        List of the potential energy of the system at each time step.
    total_energy_list : list
        List of the total energy of the system at each time step.
    """
    force, _ = compute_forces_and_potential_energy(
        position,
        box_size,
        sigma=sigma,
        epsilon=epsilon
    )
    position_list = []
    kinetic_energy_list = []
    potential_energy_list = []
    total_energy_list = []

    for step_i in range(n_steps):
        position, velocity, force = step(
            position,
            velocity,
            force,
            dt,
            box_size,
            sigma=sigma,
            epsilon=epsilon
        )

        if step_i % printing_step == 0:
            kinetic_energy = compute_kinetic_energy(velocity)
            _, potential_energy = compute_forces_and_potential_energy(
                position,
                box_size,
                sigma=sigma,
                epsilon=epsilon
            )
            kinetic_energy_list.append(kinetic_energy)
            potential_energy_list.append(potential_energy)
            total_energy_list.append(kinetic_energy + potential_energy)
            print(
                f'Step {step_i} done.\t',
                f'E_kin = {kinetic_energy}\t',
                f'E_pot = {potential_energy}',
                )

        if step_i % writing_step == 0:
            print('Saving positions')
            if center:
                position_list.append(
                    position_center_box(position,box_size=box_size)
                )
            else:
                position_list.append(position)

    return (
        jnp.array(position_list),
        kinetic_energy_list,
        potential_energy_list,
        total_energy_list
    )


@jax.jit
def compute_kinetic_energy(velocity: jnp.ndarray) -> float:
    """Compute the kinetic energy of the system.

    Parameters
    ----------
    velocity : jnp.ndarray
        Velocity of the particules.

    Returns
    -------
    float
        Kinetic energy of the system.
    """
    return jnp.sum(velocity ** 2)


def position_center_box(
        position: jnp.ndarray,
        box_size: float
    ) -> jnp.ndarray:
    """Center the position of the particules in the box.

    Parameters
    ----------
    position : jnp.ndarray
        Position of the particules.
        The shape of the array is (n, 3) where n is the number of particules.
    box_size : float
        Size of the simulation box.

    Returns
    -------
    jnp.ndarray
        Centered position of the particules.
    """
    return position - jnp.floor(position / box_size) * box_size
        

def main():
    """Run the main function."""
    parser = Parser()
    kwargs = parser.get_dynamics_kwargs()
    (
        pos_list,
        kinetic_energy_list,
        potential_energy_list,
        total_energy_list
    ) = dynamics(**kwargs)

    if parser.display_energy:
        visual.plot_energies(
            kinetic_energy_list,
            potential_energy_list,
            total_energy_list
        )

    if parser.display_animation:
        visual.animate(pos_list, parser.arguments['box_size'])

    writer.write_arc(pos_list, atom_type=parser.atom_type)


if __name__ == "__main__":

    # from jax import random

    # epsilon = 1.0
    # sigma = 1.0
    # num_particule = 10
    # box_size = 10.0
    # key = random.PRNGKey(0)

    # pos, vel = Parser.initialize_system(num_particule, box_size, key)

    # (
    #     pos_list,
    #     kinetic_energy_list,
    #     potential_energy_list,
    #     total_energy_list
    # ) = dynamics(pos, vel, 0.001, box_size, epsilon, sigma, n_steps=10000)

    # plot_energies(
    #     kinetic_energy_list,
    #     potential_energy_list,
    #     total_energy_list
    # )
    # animate(pos_list, box_size)

    main()
