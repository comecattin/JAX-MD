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
