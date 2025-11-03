# test-particle-sim-1D

A Monte Carlo test-particle simulator developed for **APC 524** — designed to model charged-particle motion and collisions in prescribed electromagnetic fields, bridging between full Particle-in-Cell (PIC) and swarm Monte Carlo (MC) approaches.


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
