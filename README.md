# grpo

This repository contains exploratory Jupyter notebooks and minimal project scaffolding to make the notebooks easy to use and reproduce.

Repository contents
- `gemma2-2b-r1-math.ipynb` — mathematical analysis and experiments (symbolic/numeric notebooks).
- `tunix.ipynb` — utilities and small experiments (UNIX/tooling demonstrations).
- `LICENSE` — MIT license.
- `.gitignore` — ignores virtualenvs, editor files, caches, and notebook checkpoints.
- `pyproject.toml` — Poetry-based project configuration for creating reproducible environments.

Libraries used
The notebooks depend on common scientific Python packages. The primary libraries used (or typically required to run the notebooks) are:

- `jupyter` / `jupyterlab`
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `sympy`

If a notebook imports additional libraries, install them via Poetry (see below) or add them to `pyproject.toml` under `[tool.poetry.dependencies]`.

Using Poetry (recommended)

1. Install Poetry (if you don't already have it):

```bash
curl -sSL https://install.python-poetry.org | python3 -
# or, alternatively:
# python3 -m pip install --user poetry
```

2. Create and activate the virtual environment managed by Poetry, and install dependencies:

```bash
cd /path/to/grpo
poetry install
poetry shell          # optional: spawn a shell inside the venv
```

3. Start Jupyter Lab from the activated environment:

```bash
poetry run jupyter lab
```

Notes on `requirements.txt`
The project now uses Poetry for dependency management and reproducible installs. `requirements.txt` has been removed in favor of `pyproject.toml`/Poetry. If you need a requirements.txt for other tooling, you can export one with:

```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

Contributing and next steps
- To add scripts or packages, consider creating a `src/` package and adding entry points to `pyproject.toml`.
- If you'd like CI (GitHub Actions), dependency pinning, or a packaged distribution, tell me which you'd prefer and I can add a starter config.

License

See the `LICENSE` file for details.
