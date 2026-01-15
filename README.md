# grpo

This repository is a small, practical setup for running GRPO-related experiments in notebooks. It’s intentionally minimal and focused on reproducibility rather than being a polished library or framework.

The notebooks center on math-style reasoning experiments, short supervised fine-tuning runs, and a single generation-based RL (GRPO) pass built on top of JAX and the `tunix` utilities. The original experiments were run on multi-device JAX setups (specifically TPU v5e-8), so reproducing them fully requires non-trivial compute.

The code relies on JAX, Flax, Optax, Orbax, and standard Hugging Face dataset tooling, along with a few project-specific helpers. Some cells assume a Kaggle-style runtime and will need to be skipped or adapted if you’re running locally.

## Environment notes

- JAX must be installed for your specific hardware (CPU, CUDA, or TPU). Follow the official installation guide:  
  https://github.com/google/jax#installation
- Some notebook cells expect Kaggle utilities (`kaggle_secrets`, `kagglehub`) and will fail outside Kaggle unless you provide equivalents.
- The notebooks install and use `google-tunix[prod]`; if you don’t have local `tunix` code, use the PyPI package referenced in the notebooks.

## Reproducible install (Poetry)

Install Poetry (if needed):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Install project dependencies:

```bash
cd /path/to/grpo
poetry install
```

## JAX installation

After installing dependencies via Poetry, install JAX for your platform.

CPU-only example:

```bash
poetry run pip install "jax[cpu]" \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

For CUDA or TPU, follow the instructions in the JAX repository.

## Running the notebooks

Launch Jupyter Lab from the Poetry environment:

```bash
poetry run jupyter lab
```

If you want to reproduce the original TPU runs, use a TPU-enabled environment (Kaggle TPU, GCP TPU, or equivalent) and configure the appropriate environment variables. Expect long runtimes and high resource usage.

## `requirements.txt`

This project uses Poetry for dependency management. If you need a `requirements.txt` for another workflow, export one with:

```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

## License

See the `LICENSE` file for details.
