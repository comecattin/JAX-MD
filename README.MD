# Molecular Dynamics using JAX

This is a simple repository to get use to JAX.

JAX-MD provide an efficient Molecular Dynamics (MD) simulation solution using [JAX](https://github.com/google/jax).

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
- `printing_step` : Print the system energy every nth step.
- `writing_step` : Write the system coordinates into a trajectory file every nth step.
- `center` : Center the molecule in the PBC box

Particles have unit mass. Reported energies use the conventional definitions
of kinetic energy (one half the sum of squared velocities) and potential
energy (each interacting pair counted once).

XYZ input accepts a standard comment line after the atom count, including an
empty comment, or the comment-free format used by the bundled example.
Randomly initialized particles use the placeholder atom label `X` in the
output trajectory.

## Generated initial conditions

When `xyz_file` is omitted, particles now start on a simple cubic lattice by
default, rather than at random positions. Set `initialization: random` to
request uniformly random positions; these can contain close particle pairs.
Both modes use `X` as the atom label. For example:

```yaml
key: 42
num_particule: 27
box_size: 6.0
initialization: lattice
temperature: 0.7
dt: 0.001
n_steps: 1000
epsilon: 1.0
sigma: 1.0
printing_step: 10
writing_step: 100
```

`temperature` defaults to 1.0 in reduced units, with particle mass and
Boltzmann constant both equal to one. Gaussian velocities have their
center-of-mass velocity removed and are rescaled so that
`sum(velocity**2) / (3*N - 3) == temperature`. This sets the initial
temperature; no thermostat is applied during dynamics. A zero temperature
gives zero velocities. A single particle requires zero temperature when net
momentum is removed. XYZ input continues to start with zero velocities.

Choose the density and timestep carefully: lattice spacing is
`box_size / ceil(N**(1/3))`, which can still be too small at high density.
The seed determines reproducible generated velocities and random positions.

The physics tests use double precision to check momentum and energy
conservation, periodic wrapping, and second-order timestep convergence.
They do not change the precision used by the command-line simulation.
