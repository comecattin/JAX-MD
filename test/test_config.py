"""Configuration boundaries and CLI failures before integration."""
import os
import subprocess
import sys

import jax.numpy as jnp
import pytest
import yaml

from jax_md.config import SimulationConfig
from jax_md.dynamics import dynamics
from jax_md.parser_md import Parser


@pytest.mark.parametrize('values, field', [
    (None, 'mapping'), ([], 'mapping'), ({}, 'mapping'),
    ({'n_step': 1}, 'n_step'),
    ({'box_size': 0}, 'box_size'), ({'box_size': True}, 'box_size'),
    ({'dt': float('nan')}, 'dt'), ({'dt': '0.01'}, 'dt'),
    ({'sigma': -1}, 'sigma'), ({'epsilon': -1}, 'epsilon'),
    ({'n_steps': 0}, 'n_steps'), ({'n_steps': 1.5}, 'n_steps'),
    ({'printing_step': 0}, 'printing_step'),
    ({'writing_step': False}, 'writing_step'),
    ({'display_energy': 'false'}, 'display_energy'),
    ({'center': 1}, 'center'), ({'temperature': float('inf')}, 'temperature'),
    ({'key': -1}, 'key'), ({'key': True}, 'key'),
    ({'num_particule': True}, 'num_particule'),
    ({'initialization': []}, 'initialization'),
    ({'xyz_file': ''}, 'xyz_file'), ({'xyz_file': 4}, 'xyz_file'),
])
def test_invalid_config(values, field, tmp_path):
    if isinstance(values, dict) and values:
        values = {'box_size': 6., 'key': 0, 'num_particule': 2, **values}
    with pytest.raises(ValueError, match=field):
        SimulationConfig.from_mapping(values, tmp_path)


@pytest.mark.parametrize('missing', ['box_size', 'key', 'num_particule'])
def test_generated_source_requires_fields(missing, tmp_path):
    values = {'box_size': 6., 'key': 0, 'num_particule': 2}
    del values[missing]
    with pytest.raises(ValueError, match=missing):
        SimulationConfig.from_mapping(values, tmp_path)


def test_null_xyz_and_defaults(tmp_path):
    config = SimulationConfig.from_mapping(
        {'box_size': 6., 'key': 0, 'num_particule': 2, 'xyz_file': None}, tmp_path)
    assert config.xyz_file is None
    assert config.dt == 0.001 and config.n_steps == 1000
    assert config.printing_step == config.writing_step == 100
    assert config.epsilon == config.sigma == 1.0
    assert not config.center and not config.display_animation


def test_relative_xyz_path_independent_of_working_directory(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'atoms.xyz').write_text('1\ncomment\nHe 1 2 3\n')
    config = data / 'run.yaml'
    config.write_text('box_size: 10\nxyz_file: atoms.xyz\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['jaxdyn', str(config)])
    parser = Parser()
    assert parser.atom_type == ['He']
    assert jnp.array_equal(parser.pos, jnp.array([[1., 2., 3.]]))


@pytest.mark.parametrize('xyz, message', [
    ('', 'empty'), ('0\n\n', 'positive integer'),
    ('wrong\n', 'atom count'), ('2\ncomment\nHe 0 0 0\n', 'count'),
    ('1\ncomment\nHe 0 0\n', 'three coordinates'),
    ('1\ncomment\nHe nan 0 0\n', 'finite'),
    ('1\ncomment\nHe 1e100 0 0\n', 'finite'),
])
def test_invalid_xyz(tmp_path, monkeypatch, xyz, message):
    (tmp_path / 'atoms.xyz').write_text(xyz)
    config = tmp_path / 'run.yaml'
    config.write_text('box_size: 10\nxyz_file: atoms.xyz\n')
    monkeypatch.setattr(sys, 'argv', ['jaxdyn', str(config)])
    with pytest.raises(ValueError, match=message):
        Parser()


@pytest.mark.parametrize('overrides, message', [
    ({'printing_step': 0}, 'printing_step'),
    ({'writing_step': 0}, 'writing_step'), ({'box_size': -1}, 'box_size'),
    ({'position': jnp.zeros((0, 3))}, 'position'),
    ({'velocity': jnp.zeros((2, 2))}, 'velocity'),
    ({'velocity': jnp.zeros((3, 3))}, 'matching'),
    ({'position': jnp.full((2, 3), jnp.nan)}, 'finite'),
    ({'velocity': jnp.full((2, 3), jnp.inf)}, 'finite'),
    ({'position': [['x', 'y', 'z']]}, 'numeric'),
    ({'velocity': jnp.ones((2, 3), dtype=complex)}, 'numeric'),
])
def test_direct_dynamics_validation(overrides, message):
    kwargs = dict(position=jnp.zeros((2, 3)), velocity=jnp.zeros((2, 3)),
                  dt=0.001, box_size=10., n_steps=1)
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        dynamics(**kwargs)


@pytest.mark.parametrize('contents, message', [
    ('box_size: [', 'Invalid YAML'), ('', 'mapping'),
    ('box_size: 10\nkey: 0\nnum_particule: 2\nwriting_step: 0\n', 'writing_step'),
    ('box_size: 10\nxyz_file: missing.xyz\n', 'missing.xyz'),
])
def test_cli_input_errors(tmp_path, contents, message):
    config = tmp_path / 'run.yaml'
    config.write_text(contents)
    result = subprocess.run(
        [sys.executable, '-m', 'jax_md.dynamics', str(config)],
        cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, 'MPLBACKEND': 'Agg'}, check=False,
    )
    assert result.returncode == 2
    assert 'jaxdyn: error:' in result.stderr
    assert message in result.stderr
    assert 'Traceback' not in result.stderr
    assert not (tmp_path / 'output.arc').exists()


def test_cli_valid_run(tmp_path):
    config = tmp_path / 'run.yaml'
    config.write_text(yaml.safe_dump(dict(
        box_size=6., key=42, num_particule=2, n_steps=2,
        printing_step=1, writing_step=1, temperature=0.,
    )))
    result = subprocess.run(
        [sys.executable, '-m', 'jax_md.dynamics', str(config)],
        cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, 'MPLBACKEND': 'Agg'}, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert len((tmp_path / 'output.arc').read_text().splitlines()) == 6
