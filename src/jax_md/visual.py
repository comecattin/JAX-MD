#!/usr/bin/env python3
"""Visualize the dynamics of the system."""
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def animate(
        pos: jnp.ndarray,
        box_size: float
        ) -> animation.FuncAnimation:
    """Animate the system.

    Parameters
    ----------
    pos : jnp.ndarray
        Position of the particules at each time step.
    box_size : float
        Size of the simulation box.

    Returns
    -------
    animation.FuncAnimation
        Animation of the system.
    """
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


def plot_energies(
        kinetic_energy_list: list,
        potential_energy_list: list,
        total_energy_list: list
    ):
    """Plot the energies of the system.

    Parameters
    ----------
    kinetic_energy_list : list
        Kinetic energy of the system at each time step.
    potential_energy_list : list
        Potential energy of the system at each time step.
    total_energy_list : list
        Total energy of the system at each time step.
    """
    plt.plot(kinetic_energy_list, label='Kinetic energy')
    plt.plot(potential_energy_list, label='Potential energy')
    plt.plot(total_energy_list, label='Total energy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    pass
