# Molecular Dynamics using JAX

This is a simple repository to get use to JAX.

JAX-MD provide an efficient Molecular Dynamics (MD) simulation solution using JAX.

Lennard-Jones potential is used as inter-atomic potential:
$$V_{ij}(r) = 4\epsilon \left[ \left( \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r} \right)^{6} \right]$$

$r$ being the inter-atomic distance between the atom $i$ and $j$, $\epsilon$ the depth of the potential well and $\sigma$ the "cutoff".

The velocity Verlet integrator is used to propagate the system along the time.

## Installation

To install JAX-MD use inside the cloned repository:

```bash
pip install -e .
```

## Usage

After the installation, to run a molecular dynamics simulation one can use:

```bash
jaxdyn param_file.yaml
```

With `param_file.yaml` being a `.yaml` file to read the simulation parameters from.

## Create an input file

JAX-MD uses `.yaml` file as input. This file contains the parameters used during the simulation. These parameters are:

- `xyz_file` : Path to the `.xyz` file containing the system.
- `key` : Key random generator, used if `xyz_file` is not given to initialize the system's position as random.
- `num_particules` : Number of particules in the randomly initialized system.
- `box_size` : Size of the simulation box.
- `dt` : Time step.
- `n_step` : Number of time step, *ie* length of the simulation.
- `epsilon` : $\epsilon$ parameter in the Lennard-Jones potential.
- `sigma` : $\sigma$ parameter in the Lennard-Jones potential.
- `display_energy` : Display energies of the system as a function of the time.
- `display_animation` : Display a 2D projection of the simulation.
