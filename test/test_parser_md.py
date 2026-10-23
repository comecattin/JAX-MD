import sys

import jax.numpy as jnp
import pytest
import yaml

from jax_md.parser_md import Parser
from jax_md.writer import write_arc


@pytest.mark.parametrize('comment', ['', '\n', 'a molecule\n'])
def test_read_xyz(tmp_path, monkeypatch, comment):
    xyz = tmp_path / 'system.xyz'
    xyz.write_text('2\n' + comment + 'He 0 1 2\nNe 3 4 5\n')
    config = tmp_path / 'input.yaml'
    config.write_text(yaml.safe_dump({'xyz_file': str(xyz)}))
    monkeypatch.setattr(sys, 'argv', ['jaxdyn', str(config)])
    parser = Parser()
    assert parser.atom_type == ['He', 'Ne']
    assert jnp.allclose(parser.pos, jnp.array([[0, 1, 2], [3, 4, 5]]))
    assert jnp.all(parser.vel == 0)


def test_random_initialization_can_write_trajectory(tmp_path, monkeypatch):
    config = tmp_path / 'input.yaml'
    config.write_text(yaml.safe_dump({
        'key': 42, 'num_particule': 3, 'box_size': 10.0,
    }))
    monkeypatch.setattr(sys, 'argv', ['jaxdyn', str(config)])
    parser = Parser()
    assert parser.atom_type == ['X'] * 3
    assert parser.pos.shape == parser.vel.shape == (3, 3)
    assert jnp.all((parser.pos >= 0) & (parser.pos < 10))
    output = tmp_path / 'output.arc'
    write_arc(parser.pos[None, ...], parser.atom_type, output)
    assert len(output.read_text().splitlines()) == 4


@pytest.mark.parametrize('mode', ['lattice', 'random'])
def test_initialization_temperature_and_reproducibility(mode):
    from jax import random

    kwargs = dict(num_particule=10, box_size=6.0, key=random.PRNGKey(4),
                  initialization=mode, temperature=0.7)
    pos, vel = Parser.initialize_system(**kwargs)
    pos_again, vel_again = Parser.initialize_system(**kwargs)
    assert jnp.array_equal(pos, pos_again)
    assert jnp.array_equal(vel, vel_again)
    # Centering, rescaling, and summing velocities introduce roundoff.
    # Bound it relative to the summed magnitudes and the actual dtype.
    tolerance = 4 * jnp.finfo(vel.dtype).eps * jnp.sum(jnp.abs(vel), axis=0)
    assert jnp.all(jnp.abs(jnp.sum(vel, axis=0)) <= tolerance)
    assert jnp.allclose(jnp.sum(vel ** 2) / (3 * 10 - 3), 0.7)
    assert jnp.all((pos >= 0) & (pos < 6.))
    if mode == 'lattice':
        displacement = pos[:, None] - pos[None, :]
        displacement -= jnp.round(displacement / 6.) * 6.
        distances = jnp.linalg.norm(displacement, axis=-1)
        distances = distances.at[jnp.diag_indices(10)].set(jnp.inf)
        assert jnp.min(distances) >= 2. - 1e-6


def test_zero_temperature_single_particle():
    from jax import random

    pos, vel = Parser.initialize_system(1, 5., random.PRNGKey(0), temperature=0.)
    assert pos.shape == (1, 3)
    assert jnp.all(vel == 0)


@pytest.mark.parametrize('overrides', [
    {'num_particule': 0}, {'num_particule': 1.5}, {'box_size': 0.},
    {'box_size': float('inf')}, {'temperature': -1.},
    {'temperature': float('nan')}, {'initialization': 'unknown'},
    {'num_particule': 1},
])
def test_invalid_initialization(overrides):
    from jax import random

    kwargs = dict(num_particule=4, box_size=5., key=random.PRNGKey(0))
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        Parser.initialize_system(**kwargs)


def test_yaml_initialization_options(tmp_path, monkeypatch):
    config = tmp_path / 'input.yaml'
    config.write_text(yaml.safe_dump({
        'key': 42, 'num_particule': 3, 'box_size': 10.0,
        'initialization': 'random', 'temperature': 0.,
    }))
    monkeypatch.setattr(sys, 'argv', ['jaxdyn', str(config)])
    parser = Parser()
    from jax import random

    expected_pos, _ = Parser.initialize_system(
        3, 10., random.PRNGKey(42), initialization='random', temperature=0.
    )
    assert jnp.array_equal(parser.pos, expected_pos)
    assert jnp.all(parser.vel == 0)
