import jax.numpy as np
import jax_md.dynamics
from jax_md.dynamics import (
    compute_forces_and_potential_energy,
    compute_kinetic_energy,
    lennard_jones,
)


def test_lennard_jones():
    # Test case 1: r = 1.0, epsilon = 1.0, sigma = 1.0
    r1 = np.array([[1.0]])
    expected1 = np.array([[0.0]])
    assert np.allclose(lennard_jones(r1), expected1)

    # Test case 2: r = 2.0, epsilon = 2.0, sigma = 2.0
    r2 = np.array([[2.0]])
    expected2 = np.array([[0.0]])
    assert np.allclose(lennard_jones(r2, epsilon=2.0, sigma=2.0), expected2)


def test_energy_conservation():
    position = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocity = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    (
        position_list,
        kinetic_energy_list,
        potential_energy_list,
        total_energy_list
    ) = jax_md.dynamics.dynamics(
        position=position,
        velocity=velocity,
        box_size=10.0,
        dt=0.001,
        n_steps=1000
    )
    initial_energy = compute_kinetic_energy(velocity) + (
        compute_forces_and_potential_energy(position, 10.0)[1]
    )
    energies = np.array(total_energy_list)
    assert np.all(np.isfinite(energies))
    assert np.max(np.abs(energies - initial_energy)) < 1e-3
    assert np.max(np.array(kinetic_energy_list)) > 0.1


def test_kinetic_energy_unit_masses():
    velocity = np.array([[1.0, 2.0, 2.0], [0.0, 0.0, 1.0]])
    assert np.allclose(compute_kinetic_energy(velocity), 5.0)


def test_pair_energy_and_forces():
    position = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    force, energy = compute_forces_and_potential_energy(position, 10.0)
    assert np.allclose(energy, lennard_jones(1.5))
    assert np.allclose(force[0], -force[1])
    delta = 1e-3
    plus = position.at[0, 0].add(delta)
    minus = position.at[0, 0].add(-delta)
    derivative = (
        compute_forces_and_potential_energy(plus, 10.0)[1]
        - compute_forces_and_potential_energy(minus, 10.0)[1]
    ) / (2 * delta)
    assert np.allclose(force[0, 0], -derivative, rtol=1e-3)



if __name__ == "__main__":
    pass
