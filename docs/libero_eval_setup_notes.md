# LIBERO Evaluation Setup Notes

These notes capture the issues encountered while setting up LIBERO evaluation on this machine on 2026-05-08.
They are meant as a troubleshooting companion to `examples/libero/README.md`.

## Environment Split

Do not try to run the OpenPI policy server and LIBERO simulator in one Python environment.

- OpenPI requires Python `>=3.11` from the root `pyproject.toml`.
- The LIBERO example requirements are locked for Python `3.8` and old simulation dependencies.

Use two environments:

```bash
mamba create -y -n openpi_server_eval python=3.11.9
mamba create -y -n openpi_libero_eval python=3.8
```

Install OpenPI into the server environment using the repo lockfile:

```bash
eval "$(conda shell.bash hook)"
conda activate openpi_server_eval
export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"
export UV_LINK_MODE=copy
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev
```

Install LIBERO runtime dependencies into the simulator environment:

```bash
eval "$(conda shell.bash hook)"
conda activate openpi_libero_eval
uv pip sync --python "$CONDA_PREFIX/bin/python" \
  examples/libero/requirements.txt \
  third_party/libero/requirements.txt \
  packages/openpi-client/pyproject.toml \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match

uv pip install --python "$CONDA_PREFIX/bin/python" --no-deps \
  -e packages/openpi-client \
  -e third_party/libero
```

`third_party/libero` was not importable from the editable install alone in this setup, so keep `PYTHONPATH` explicit:

```bash
export PYTHONPATH="$PWD/third_party/libero:$PWD/packages/openpi-client/src:$PYTHONPATH"
```

## LIBERO Config Path

Create a LIBERO config file to avoid interactive initialization and to point at the vendored submodule:

```bash
mkdir -p /tmp/libero
cat > /tmp/libero/config.yaml <<EOF
benchmark_root: $PWD/third_party/libero/libero/libero
bddl_files: $PWD/third_party/libero/libero/libero/bddl_files
init_states: $PWD/third_party/libero/libero/libero/init_files
datasets: $PWD/third_party/libero/libero/datasets
assets: $PWD/third_party/libero/libero/libero/assets
EOF
export LIBERO_CONFIG_PATH=/tmp/libero
```

The warning about `third_party/libero/libero/datasets` not existing did not block environment reset for benchmark evaluation, because benchmark init states and assets are included in the submodule.

## Checkpoint And Tokenizer Downloads

The LIBERO checkpoint is:

```bash
gs://openpi-assets/checkpoints/pi05_libero
```

In this environment, `gcsfs` failed to resolve `storage.googleapis.com` from Python, while `curl` and `gsutil` worked through the shell proxy. Install `gsutil` in the server environment and let OpenPI's `openpi-assets` path use it:

```bash
conda activate openpi_server_eval
python -m pip install gsutil
python - <<'PY'
from openpi.shared import download
print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero"))
PY
```

The checkpoint was cached at:

```text
~/.cache/openpi/openpi-assets/checkpoints/pi05_libero
```

Policy creation also needs the PaliGemma tokenizer from `gs://big_vision/paligemma_tokenizer.model`.
That path falls back to `gcsfs`, so if Python DNS/proxy resolution fails, download it manually:

```bash
mkdir -p ~/.cache/openpi/big_vision
gsutil cp gs://big_vision/paligemma_tokenizer.model \
  ~/.cache/openpi/big_vision/paligemma_tokenizer.model
```

## Avoid User-Site Package Leakage

Set `PYTHONNOUSERSITE=1` when running the OpenPI server. Without this, Python imported packages from `~/.local/lib/python3.11/site-packages` instead of only the conda environment, which made dependency behavior unreliable.

```bash
export PYTHONNOUSERSITE=1
```

## Server-Side Segfaults

Two inference-only import paths pulled in training or PyTorch dependencies and caused native segfaults in this setup:

- `openpi.policies.policy` imported `torch` at module import time.
- `openpi.policies.policy_config` imported `openpi.training.checkpoints`, which imports training data loader code and `torch`.

The local fix was:

- Move `torch` imports in `src/openpi/policies/policy.py` into PyTorch-only branches.
- Load norm stats in `src/openpi/policies/policy_config.py` via `openpi.shared.normalize.load()` instead of importing `openpi.training.checkpoints`.

After those changes, creating the `pi05_libero` JAX policy completed successfully.

## Rendering Backend

The README defaults to EGL, but EGL failed in this shell:

```text
RuntimeError: The MUJOCO_EGL_DEVICE_ID environment variable must be an integer between 0 and -1 (inclusive), got 0.
ImportError: Cannot initialize a EGL device display.
```

This machine also had no usable `DISPLAY`, so GLX was not available from the shell. OSMesa worked:

```bash
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
unset MUJOCO_EGL_DEVICE_ID
```

Smoke test used:

```bash
conda activate openpi_libero_eval
export PYTHONPATH="$PWD/third_party/libero:$PWD/packages/openpi-client/src:$PYTHONPATH"
export LIBERO_CONFIG_PATH=/tmp/libero
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
python - <<'PY'
import pathlib
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=64, camera_widths=64)
obs = env.reset()
print(obs["agentview_image"].shape)
env.close()
PY
```

Expected output includes:

```text
(64, 64, 3)
```

## Run Commands

Start the server:

```bash
conda activate openpi_server_eval
export PYTHONNOUSERSITE=1
export OPENPI_DATA_HOME="$HOME/.cache/openpi"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -u scripts/serve_policy.py --env LIBERO --port 8000
```

Verify the server:

```bash
curl -fsS http://127.0.0.1:8000/healthz
```

Run a small smoke evaluation:

```bash
conda activate openpi_libero_eval
export PYTHONPATH="$PWD/third_party/libero:$PWD/packages/openpi-client/src:$PYTHONPATH"
export LIBERO_CONFIG_PATH=/tmp/libero
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
python -u examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1 \
  --args.video-out-path data/libero/videos_smoke
```

The attempted smoke eval was stopped before completion at user request. The server had previously reached `server listening on 0.0.0.0:8000`, and the simulator reset smoke test passed with OSMesa.
