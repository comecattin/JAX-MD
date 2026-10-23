"""Physical invariants and timestep convergence in double precision."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64

from jax_md.dynamics import (
    compute_forces_and_potential_energy,
    compute_kinetic_energy,
    step,
)


@pytest.fixture
def double_precision():
    with enable_x64():
        yield


def energy(position, velocity, box):
    return compute_kinetic_energy(velocity) + (
        compute_forces_and_potential_energy(position, box)[1]
    )


def trajectory(dt, duration=0.5):
    # A moving, interacting cluster crosses the periodic x boundary.
    position = jnp.array([[5.8, 2., 2.], [0.95, 2., 2.],
                          [5.8, 3.2, 2.], [5.8, 2., 3.25]])
    velocity = jnp.array([[0.7, 0.1, 0.], [0.6, -0.1, 0.],
                          [0.7, 0., 0.1], [0.8, 0., -0.1]])
    force, _ = compute_forces_and_potential_energy(position, 6.)
    initial = energy(position, velocity, 6.)
    momentum = jnp.sum(velocity, axis=0)

    def advance(state, _):
        p, v, f = step(*state, dt, 6.)
        return (p, v, f), (p, energy(p, v, 6.), jnp.sum(v, axis=0))

    _, (positions, energies, momenta) = jax.lax.scan(
        advance, (position, velocity, force), None, length=round(duration / dt)
    )
    return np.asarray(positions), np.asarray(energies - initial), (
        np.asarray(momenta - momentum)
    )


def test_many_particle_conservation_and_boundary_crossing(double_precision):
    positions, drift, momentum_drift = trajectory(0.001)
    assert np.isfinite(drift).all()
    assert np.max(np.abs(drift)) < 1e-4
    assert np.max(np.abs(momentum_drift)) < 1e-12
    assert np.all((positions >= 0) & (positions < 6.))
    assert np.any(np.diff(positions[:, 0, 0]) < -5.)


def test_verlet_second_order_energy_error(double_precision):
    errors = [np.max(np.abs(trajectory(dt)[1]))
              for dt in (0.004, 0.002, 0.001)]
    # Halving dt should reduce the bounded energy error by approximately four.
    assert 3.5 < errors[0] / errors[1] < 4.5
    assert 3.5 < errors[1] / errors[2] < 4.5


def test_periodic_force_and_energy_invariance(double_precision):
    position = jnp.array([[0.1, 1., 1.], [5., 1.2, 1.], [2., 2., 2.]])
    force, potential = compute_forces_and_potential_energy(position, 6.)
    shifted = position.at[0].add(jnp.array([6., -6., 12.]))
    shifted_force, shifted_potential = compute_forces_and_potential_energy(shifted, 6.)
    np.testing.assert_allclose(shifted_force, force, atol=1e-12)
    np.testing.assert_allclose(shifted_potential, potential, atol=1e-12)
    np.testing.assert_allclose(jnp.sum(force, axis=0), 0., atol=1e-12)
