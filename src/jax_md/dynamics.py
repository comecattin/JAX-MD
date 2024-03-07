#!/usr/bin/env python3
"""Molecular dynamics."""
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from .parser import Parser


@jax.jit
def lennard_jones(r, epsilon=1.0, sigma=1.0):
    """Lennard-Jones potential."""
    r6 = (sigma / r) ** 6
    r12 = r6 ** 2
    return 4.0 * epsilon * (r12 - r6)


@jax.jit
def compute_forces_and_potential_energy(pos, box_size, epsilon=1.0, sigma=1.0):
    """Compute the forces and the potential energy."""
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


def step(position, velocity, force, dt, box_size, epsilon=1.0, sigma=1.0):
    """Update the system using the velocity Verlet algorithm."""
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
        position,
        velocity,
        dt,
        box_size,
        epsilon=1.0,
        sigma=1.0,
        n_steps=1000
    ):
    """Run the dynamics."""
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
        position_list.append(position)
        if step_i % 100 == 0:
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
    return (
        jnp.array(position_list),
        kinetic_energy_list,
        potential_energy_list,
        total_energy_list
    )


@jax.jit
def compute_kinetic_energy(velocity):
    """Compute the kinetic energy."""
    return jnp.sum(velocity ** 2)


def plot_energies(
        kinetic_energy_list,
        potential_energy_list,
        total_energy_list
    ):
    """Plot the energies."""
    plt.plot(kinetic_energy_list, label='Kinetic energy')
    plt.plot(potential_energy_list, label='Potential energy')
    plt.plot(total_energy_list, label='Total energy')
    plt.legend()
    plt.show()


def animate(pos, box_size):
    """Animate the system."""
    fig, ax = plt.subplots()
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    sc = ax.scatter(pos[:, 0], pos[:, 1])

    def update(frame):
        sc.set_offsets(pos[frame][:, :2])
        return sc,

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=pos.shape[0],
        blit=True,
        interval=1,
    )
    plt.show()
    return ani

def main():
    parser = Parser()
    kwargs = parser.get_dynamics_kwargs()
    (
        pos_list,
        kinetic_energy_list,
        potential_energy_list,
        total_energy_list
    ) = dynamics(**kwargs)
    plot_energies(kinetic_energy_list, potential_energy_list, total_energy_list)
    animate(pos_list, parser.arguments['box_size'])


if __name__ == "__main__":
    # epsilon = 1.0
    # sigma = 1.0
    # num_particule = 10
    # box_size = 10.0
    # key = random.PRNGKey(0)
    # pos, vel = initialize_system(num_particule, box_size, key)
    # (
    #     pos_list,
    #     kinetic_energy_list,
    #     potential_energy_list,
    #     total_energy_list
    # ) = dynamics(pos, vel, 0.001, box_size, epsilon, sigma, n_steps=10000)
    # plot_energies(kinetic_energy_list, potential_energy_list, total_energy_list)
    # animate(pos_list, box_size)
    main()
