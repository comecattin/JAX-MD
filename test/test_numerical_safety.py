"""Numerical safety in float32, including periodic overlaps."""
import jax.numpy as jnp
import pytest

from jax_md import dynamics as md


def run(position, **kwargs):
    return md.dynamics(
        position, jnp.zeros_like(position), box_size=10., dt=0.001,
        n_steps=2, **kwargs,
    )


def test_float32_self_interactions_are_safe():
    position = jnp.zeros((1, 3), dtype=jnp.float32)
    force, energy = md.compute_forces_and_potential_energy(position, 10.)
    assert jnp.all(force == 0)
    assert energy == 0


@pytest.mark.parametrize('distance', [0., 10., -20.])
def test_reject_initial_periodic_overlap(distance):
    position = jnp.array([[0., 0., 0.], [distance, 0., 0.]],
                         dtype=jnp.float32)
    with pytest.raises(ValueError, match='particles 0 and 1.*overlap'):
        run(position)


def test_reject_unrepresentable_close_pair():
    position = jnp.array([[0., 0., 0.], [1e-6, 0., 0.]],
                         dtype=jnp.float32)
    with pytest.raises(ValueError, match='Nonfinite.*step 0.*sigma='):
        run(position)


def test_close_pair_uses_unmodified_potential():
    position = jnp.array([[0., 0., 0.], [0.5, 0., 0.]],
                         dtype=jnp.float32)
    force, energy = md.compute_forces_and_potential_energy(position, 10.)
    assert jnp.all(jnp.isfinite(force))
    assert jnp.isclose(energy, md.lennard_jones(0.5))


def test_stop_before_saving_nonfinite_state(monkeypatch, capsys):
    def broken_step(position, velocity, force, *args, **kwargs):
        return position, velocity.at[0, 0].set(jnp.inf), force

    monkeypatch.setattr(md, 'step', broken_step)
    with pytest.raises(ValueError, match='Nonfinite.*step 1.*time 0.001'):
        run(jnp.array([[0., 0., 0.], [1.5, 0., 0.]]),
            printing_step=100, writing_step=100)
    assert 'Saving positions' not in capsys.readouterr().out
