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
- `num_particule` : Number of particules in the randomly initialized system.
- `box_size` : Size of the simulation box.
- `dt` : Time step.
- `n_steps` : Number of time step, *ie* length of the simulation.
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

## Input validation and defaults

Configuration is validated before reading coordinates or generating particles.
`box_size` is required. Supply either `xyz_file` or both `key` and
`num_particule`; omitting `xyz_file` or setting it to `null` selects generated
initial conditions. Unknown fields are rejected, including misspelled names.

Relative `xyz_file` paths are resolved from the directory containing the YAML
file, independent of the working directory. Absolute paths are also accepted.
For example, from the repository root:

```bash
jaxdyn example/data/input_file/input.yaml
```

The output is still written to `output.arc` in the working directory.

Omitted optional fields use these defaults:

| Field | Default | Constraint |
| --- | --- | --- |
| `dt` | `0.001` | Positive, finite number |
| `n_steps` | `1000` | Positive integer |
| `epsilon` | `1.0` | Nonnegative, finite number |
| `sigma` | `1.0` | Positive, finite number |
| `printing_step` | `100` | Positive integer |
| `writing_step` | `100` | Positive integer |
| `center` | `false` | Boolean |
| `display_energy` | `false` | Boolean |
| `display_animation` | `false` | Boolean |
| `initialization` | `lattice` | `lattice` or `random` |
| `temperature` | `1.0` | Nonnegative, finite number |

`box_size` must be positive and finite, `num_particule` a positive integer,
and `key` an integer from 0 through 4294967295. Booleans are not accepted as
numbers. YAML booleans must be unquoted (`false`, rather than `"false"`).
Initialization options are validated even when XYZ input is selected; XYZ
particles continue to start with zero velocity.

XYZ input must have a positive atom count, matching coordinate rows, and three
finite numeric coordinates per atom. The public `dynamics()` function also
validates parameters and matching, nonempty `(N, 3)` position/velocity arrays
before integration. Zero-step runs and zero output intervals are rejected.
Invalid CLI input exits with status 2 and a field-specific error on stderr.
Particle overlap checks and detection of numerical failures during integration
are tracked separately in issue #16.
