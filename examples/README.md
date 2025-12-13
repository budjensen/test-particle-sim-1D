# Examples

This directory contains demonstration scripts showcasing the capabilities of `test-particle-sim-1D`. Each example simulates charged-particle dynamics in different electromagnetic field configurations with optional Monte Carlo collisions (MCC).

---

## Directory Structure

Each example follows a consistent layout:
```
example_name/
├── script.py               # Main simulation script
├── plotting/               # Visualization scripts
│   └── plotting.py
└── results/                # Output data (generated after running)
    ├── *.npz              # Full simulation state
    ├── *.csv              # Diagnostic time-series
    └── tracer_*.csv       # Particle trajectories
```

---

## Available Examples

### 1. **uniformE** - Uniform Electric Field
**Script:** [`uniformE.py`](uniformE/uniformE.py)

Ions in a uniform electric field with elastic collisions and periodic boundary conditions.

**Physics demonstrated:**
- Electric field acceleration
- Elastic scattering with neutrals
- Drift velocity
- Energy thermalization

---

### 2. **uniformE_B** - Crossed E and B Fields
**Script:** [`uniformE_B.py`](uniformE_B/uniformE_B.py)

Ions in crossed uniform electric and magnetic fields, demonstrating **ExB drift** and gyromotion.

**Physics demonstrated:**
- ExB drift perpendicular to both fields
- Cyclotron motion (gyromotion)

---

### 3. **magneticmirror** - Magnetic Mirror Confinement
**Script:** [`magneticmirror.py`](magneticmirror/magneticmirror.py)

Ions in a spatially-varying magnetic field exhibiting **magnetic mirror confinement**.

**Physics demonstrated:**
- Magnetic mirror force from field gradient
- Conservation of first adiabatic invariant (magnetic moment)
- Trapped vs. passing particle trajectories

---

### 4. **child_law_sheath** - Collisional and Collisionless Sheaths
**Scripts:**
- [`collisionless_child_law_script.py`](child_law_sheath/collisionless_child_law_script.py)
- [`collisional_child_law_script.py`](child_law_sheath/collisional_child_law_script.py)

Ions traversing a **Child-Langmuir sheath** near a plasma-wall boundary with absorbing wall conditions.

**Physics demonstrated:**
- Child-Langmuir sheath electric field: `E(z) ∝ z^(1/3)`
- Ion acceleration through sheath
- **Ion energy distribution function (IEDF)** at the wall
- Effect of collisions on IEDF (energy loss, scattering)

---

## Running Examples

### Basic Workflow

1. **Activate the examples dependency group:**
    ```bash
    uv sync --group examples
    ```

1. **Navigate to desired example and run the simulation:**
   ```bash
   cd examples/uniformE
   uv run uniformE.py
   ```

1. **Generate plots:**
   ```bash
   uv run plotting/uniformE_plotting.py
   ```

1. **View results:**
   Output files are saved in `results/`
   and plots in `plotting/`

---

## Further Examples

- Physics tests in [`tests/`](../tests/)

---

## Contributing

To add a new example:
1. Create a new directory under `examples/` with a simulation script
1. Create `plotting/` subdirectory with visualization script
1. Document physics and parameters in this README
