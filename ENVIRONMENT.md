# ENVIRONMENT

## Python

Recommended Python version:

- Python `3.10`

Install with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r docker/requirements.txt
```

## Conda

```bash
conda env create -f docker/environment.yml
conda activate smarttalk-artifact
```

## Docker

Build:

```bash
docker build -t smarttalk-artifact -f docker/Dockerfile .
```

## GPU Use

SMARTTalk offline pattern learning and some baselines benefit strongly from GPUs.

- CNN / autoencoder training: use `--device cuda:0`
- multi-GPU local vLLM serving: use `CUDA_VISIBLE_DEVICES=...`

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash smarttalk/_legacy/scripts/serve_vllm.sh microsoft/phi-4 Phi-4 --tensor-parallel-size 4 --port 8000
```

## Live LLM Inference

You can run with:

- OpenAI-compatible cloud endpoints, or
- local vLLM endpoints.

Common arguments:

- `--base-url`
- `--api-key`
- `--model-name`

## No-API-Key Mode

Cached reproduction does not require API keys:

```bash
bash scripts/07_reproduce/reproduce_from_cache.sh
```

## Important Python Packages

- `numpy`
- `pandas`
- `scikit-learn`
- `torch`
- `openai`
- `vllm`
- `matplotlib`
- `seaborn`
- `openpyxl`
- `PyYAML`
- `pytest`

## Notes

- This artifact intentionally separates cached reproduction from live inference.
- Full processed split trees are not included in the repository; rebuild them
  locally with the provided preprocessing scripts when needed.
