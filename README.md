# grpo

This repository contains exploratory Jupyter notebooks and minimal project scaffolding to make the notebooks easy to use and reproduce.

Repository contents
- `gemma2-2b-r1-math.ipynb` — mathematical analysis and experiments (symbolic/numeric notebooks).
- `tunix.ipynb` — utilities and small experiments (UNIX/tooling demonstrations).
- `LICENSE` — MIT license.
- `.gitignore` — ignores virtualenvs, editor files, caches, and notebook checkpoints.
- `pyproject.toml` — Poetry-based project configuration for creating reproducible environments.

What each notebook does

- `gemma2-2b-r1-math.ipynb` — End-to-end experiment that prepares GSM8K-style data, runs a short supervised fine-tuning (SFT) loop and a single GRPO (generation-based RL) job using the `tunix` training utilities. This notebook expects access to a multi-device JAX runtime (the original runs target TPU v5e-8 and assert `jax.device_count() == 8`). It downloads or reads datasets (GSM8K), constructs training datasets via `grain` helpers, creates optimizers with `optax`, checkpoints with `orbax`, and saves a final ZIP of model checkpoints.

- `tunix.ipynb` — Collection of helper and demo code used by the `tunix` tooling: dataset access, model parameter helpers, samplers, and utility wrappers. It demonstrates downloading snapshots from the Hugging Face Hub, interacting with `kagglehub`, and simple data-processing helpers used by the other notebook.

Exact libraries used (extracted from the notebooks)

Primary runtime and ML libraries:

- `jax` — numerical backend used for all model execution (TPU/GPU/CPU-specific builds required).
- `flax` — model library used by `tunix` models (`nnx` usage appears in the notebooks).
- `optax` — optimizers.
- `orbax` — checkpointing utilities.

Data / hub / dataset libraries:

- `datasets` (Hugging Face `datasets`) — dataset loading and preprocessing.
- `huggingface-hub` — snapshot downloads, login utilities.
- `kagglehub` — dataset helper used in the notebooks; some dataset access uses Kaggle-specific helpers.
- `tensorflow-datasets` — used in `tunix.ipynb` for example dataset loading.

Utility libraries:

- `numpy`, `pandas`, `tqdm`, `humanize`, `rapidfuzz`, `safetensors`.
- `grain`, `qwix` — project-specific helper packages referenced by the notebooks.
- `google-tunix` — the `tunix` package used by the notebooks; the notebooks install `google-tunix[prod]==0.1.3` in a cell.

Notes about environment and installation

- These notebooks have been developed to run on a host with JAX configured for multiple devices. `jax`/`jaxlib` must be installed according to your hardware (CPU, CUDA, or TPU). See https://github.com/google/jax#installation for platform-specific instructions.
- Some cells expect Kaggle runtime utilities (for example `kaggle_secrets` and `kagglehub`) and will fail outside Kaggle unless you provide equivalent secrets or skip those cells.
- `tunix` is referenced as an installed package in the notebooks (`google-tunix[prod]`); if you do not have a local `tunix` package, install the PyPI distribution used by the author or adjust the import paths to point to local code.

Reproducible install (Poetry)

1. Install Poetry (if you don't already have it):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install the project dependencies:

```bash
cd /path/to/grpo
poetry install
```

3. Notes for JAX / TPU / GPU users

- After `poetry install`, install `jax`/`jaxlib` appropriate for your platform. Example for CPU-only:

```bash
poetry run pip install "jax[cpu]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

- For CUDA or TPU follow the instructions on the JAX repo linked above.

Running the notebooks

- Start Jupyter Lab from the Poetry environment:

```bash
poetry run jupyter lab
```

- If you plan to reproduce the original TPU runs, run on a TPU-enabled environment (Kaggle TPU, GCP TPU, or equivalent) and set the appropriate environment variables. Expect long runtimes and significant resource usage.

If you need a `requirements.txt` for a different workflow, generate it from Poetry:

```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

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

- The notebooks currently depend on `tunix` and several infrastructure packages that may not be available on plain CPython installs. If you want, I can:
	- Add a minimal `src/` layout and local editable install for `tunix`.
	- Add CI that checks notebook execution on a small sample (no TPU required).
	- Pin exact working versions for reproducibility.

License

See the `LICENSE` file for details.
