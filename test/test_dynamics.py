import jax.numpy as np
import jax_md.dynamics
from jax_md.dynamics import lennard_jones


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
    assert np.allclose(
        np.array(kinetic_energy_list) + np.array(potential_energy_list),
        np.array(total_energy_list)
    )


if __name__ == "__main__":
    pass
