#!/usr/bin/env python3
"""Write the output of the dynamic."""
import jax.numpy as jnp


def write_arc(
        pos_list: jnp.ndarray,
        atom_type: list[str],
        output_file: str = 'output.arc'
    ):
    """Write the output of the dynamic in a .arc file.

    Parameters
    ----------
    pos_list : jnp.ndarray
        Position of the particles at each step.
    atom_type : list[str]
        Atom elements.
    output_file : str, optional
        Path to the output file, by default 'output.arc'
    """
    num_particles = pos_list[0].shape[0]
    with open(output_file, 'w') as file:
        for position in pos_list:
            file.write(f'{num_particles}\n')
            for i, atom in enumerate(position):
                x, y, z = atom
                type_of_atom = atom_type[i]
                file.write(f'{i + 1} {type_of_atom} {x} {y} {z}\n')


if __name__ == "__main__":
    pass
