# test-particle-sim-1D

A Monte Carlo **test-particle simulator** developed for **APC 524**.
This codebase models **charged-particle motion in 1D3V** (1 spatial dimension, 3 velocity components) under prescribed electromagnetic fields and performs **Monte Carlo collisions (MCC)** with background neutrals.

The project bridges between **Particle-in-Cell (PIC)** and **swarm Monte Carlo (MC)** methods:
- Particles are pushed using the **Boris integrator** under user-defined E and B fields
- Collisions use a **null-collision MCC model** with support for elastic scattering, ionization, charge exchange, and excitation
- Diagnostics compute temperature, drift, density profiles, and energy distributions
- Example scripts demonstrate how to run simulations, save results, and generate plots

This repository includes:
- A full MCC collision engine
- A modular particle pusher
- Diagnostic tools
- Example simulations (`example_uniformE`, etc.)
- Automated plotting utilities
- Pre-commit formatting, linting, and testing
- A growing test suite validating physics (isotropy, thermalization, distribution convergence, etc.)

---

## Features

### **Physics Modules**
- **Boris pusher** for charged-particle motion in E and B fields
- **Electromagnetic field models:**
  - Uniform electric field
  - Uniform magnetic field
  - Magnetic mirror configuration
  - Space-dependent fields via user-defined functions
- **Monte Carlo Collision (MCC) engine:**
  - Null-collision method for efficiency
  - Elastic scattering
  - Ionization (with product species creation)
  - Charge exchange
  - Excitation collisions
  - Energy-dependent cross-sections
  - Isotropic scattering in center-of-mass frame

### **Diagnostics**
- Global temperature
- Component temperatures (Tx, Ty, Tz)
- Drift velocity
- Density profiles (`n(z)`)
- Temperature profiles (`T(z)`)
- Energy distribution functions (EEDF/IEDF)
- Particle trajectory recording


## Installation (with [uv](https://github.com/astral-sh/uv))

`uv` is a fast Python package/environment manager developed by Astral.

### 1. Install `uv`
If you don’t already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your shell (or run `source ~/.bashrc` / `source ~/.zshrc`).

Check installation:
```bash
uv --version
```

---

### 2. Clone the repository

```bash
git clone https://github.com/your-username/test-particle-sim-1D.git
cd test-particle-sim-1D
```

---

### 3. Create and sync environment

```bash
uv sync
```

This installs all dependencies defined in `pyproject.toml`.

To activate the environment:
```bash
uv run python
```
or for any command:
```bash
uv run <command>
```

Example:
```bash
uv run pytest -v
```

## Development Tools

### Run linters and formatters

```bash
uv run pre-commit run --all-files
```

The pre-commit hooks include:
- **ruff** for linting and formatting
- **nbstripout** for cleaning notebooks
- Basic style checks (whitespace, merge conflicts, etc.)

### Run tests

```bash
uv run pytest -v
```
