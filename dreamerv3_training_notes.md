# DreamerV3 RTX 5090 Training Notes

Date: 2026-06-15
Host user/workdir: `/home/ofeai`, `/home/ofeai/DreamV3`

## Environment

- GPU: NVIDIA GeForce RTX 5090
- VRAM: 32607 MiB
- Driver: 580.159.03
- CUDA reported by driver: 13.0
- Disk: `/dev/nvme0n1p2`, 1.8T total, 235G used, 1.5T available
- Conda root: `/home/ofeai/miniconda3`
- Conda env: `DreamV3`
- Python: 3.11.15
- JAX: 0.10.1
- jaxlib: 0.10.1
- NumPy: 2.4.6
- JAX devices: `[CudaDevice(id=0)]`

Note: The requested `/root/autodl-tmp` path is not accessible to the current `ofeai` user (`/root` is permission denied). Logs were written to the equivalent user-writable path:

```bash
/home/ofeai/autodl-tmp/logdir
```

## Installation Steps Used

Created the Python 3.11 environment:

```bash
/home/ofeai/miniconda3/bin/conda create -y -n DreamV3 python=3.11 pip
```

Downloaded DreamerV3 source from GitHub as a zip because system `git` was not available:

```bash
curl -L https://github.com/danijar/dreamerv3/archive/refs/heads/main.zip -o /tmp/dreamerv3-main.zip
unzip -q /tmp/dreamerv3-main.zip -d /tmp/dreamerv3-src
cp -a /tmp/dreamerv3-src/dreamerv3-main/. /home/ofeai/DreamV3/
```

Installed JAX CUDA 13 for Blackwell/RTX 5090. The default PyPI download repeatedly stalled on `jax-cuda13-pjrt`, so `uv` with the Tsinghua mirror was used:

```bash
/home/ofeai/.local/bin/uv pip install \
  --python /home/ofeai/miniconda3/envs/DreamV3/bin/python \
  -U "jax[cuda13]" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Installed remaining DreamerV3 dependencies from `requirements.txt`, skipping old CUDA/JAX pins and also skipping `numpy<2` because JAX 0.10.1 requires NumPy >= 2:

Skipped:

```text
jax[cuda12]==0.4.33
nvidia-cuda-nvcc-cu12<=12.2
numpy<2
```

Installed filtered dependencies plus Crafter:

```bash
awk '!/^jax\[cuda12\]==/ && !/^nvidia-cuda-nvcc-cu12/ && !/^numpy<2/' \
  requirements.txt > /tmp/dreamerv3_requirements_filtered.txt

/home/ofeai/.local/bin/uv pip install \
  --python /home/ofeai/miniconda3/envs/DreamV3/bin/python \
  -r /tmp/dreamerv3_requirements_filtered.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

/home/ofeai/.local/bin/uv pip install \
  --python /home/ofeai/miniconda3/envs/DreamV3/bin/python \
  crafter \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Validated JAX:

```bash
/home/ofeai/miniconda3/envs/DreamV3/bin/python - <<'PY'
import jax
print(jax.__version__)
print(jax.devices())
PY
```

Output:

```text
0.10.1
[CudaDevice(id=0)]
```

## Compatibility Fixes Applied

DreamerV3 source uses older `jax.jit()` positional arguments. JAX 0.10 requires sharding and donate arguments to be passed by keyword. The following files were patched:

- `embodied/jax/transform.py`
- `embodied/jax/internal.py`
- `embodied/jax/agent.py`

Examples of the fix:

```python
jax.jit(fn, in_shardings=..., out_shardings=..., static_argnums=...)
jax.jit(fn, in_shardings=..., out_shardings=..., donate_argnums=...)
```

Also, `--configs debug` sets `jax.platform: cpu`, so debug training must explicitly override it with:

```bash
--jax.platform cuda
```

## Debug Training

Actual command used:

```bash
/home/ofeai/miniconda3/envs/DreamV3/bin/python dreamerv3/main.py \
  --logdir /home/ofeai/autodl-tmp/logdir/dreamer/debug_crafter \
  --configs crafter debug \
  --run.train_ratio 32 \
  --jax.platform cuda
```

Result: passed. It initialized on `cuda:0`, compiled, saved checkpoints, wrote metrics, and reached Agent Step 8640 before being stopped to free the GPU for formal training.

Observed debug metrics included:

```text
JAX devices (1): [cuda:0]
Policy devices: cuda:0
Train devices:  cuda:0
fps/policy ~180-220
fps/train ~5600-7100
```

## Formal Training

Actual command, running in background:

```bash
cd /home/ofeai/DreamV3
setsid /home/ofeai/miniconda3/envs/DreamV3/bin/python dreamerv3/main.py \
  --logdir /home/ofeai/autodl-tmp/logdir/dreamer/crafter_size50m_5090 \
  --configs crafter size50m \
  --run.train_ratio 32 \
  --jax.platform cuda \
  >> /home/ofeai/autodl-tmp/logdir/dreamer/crafter_size50m_5090/train_stdout.log 2>&1 < /dev/null &
```

Current Python process:

```text
PID: 782316
Command: dreamerv3/main.py --configs crafter size50m --run.train_ratio 32 --jax.platform cuda
```

Initial formal run status:

- JAX devices: `[cuda:0]`
- Model params: 41,658,259
- GPU memory during training: about 25.8 GiB used out of 32.6 GiB
- First checkpoint written under:

```bash
/home/ofeai/autodl-tmp/logdir/dreamer/crafter_size50m_5090/ckpt
```

- Metrics written under:

```bash
/home/ofeai/autodl-tmp/logdir/dreamer/crafter_size50m_5090/metrics.jsonl
/home/ofeai/autodl-tmp/logdir/dreamer/crafter_size50m_5090/scores.jsonl
```

Sample formal metric:

```text
Agent Step 12_260
replay/replay_ratio 29.93
fps/policy 100.68
fps/train 2943.18
```

## Monitoring

Check process:

```bash
ps -p 782316 -o pid,ppid,stat,etime,pcpu,pmem,rss,cmd
```

Check GPU:

```bash
nvidia-smi
```

Follow stdout/stderr:

```bash
tail -f /home/ofeai/autodl-tmp/logdir/dreamer/crafter_size50m_5090/train_stdout.log
```

Follow scalar metrics:

```bash
tail -f /home/ofeai/autodl-tmp/logdir/dreamer/crafter_size50m_5090/metrics.jsonl
```

Stop training if needed:

```bash
kill 782316
```

## Scope Viewer

`scope==0.7.0` is installed. In this version, the available entry point is `scope`, not `python -m scope.viewer`.

Start viewer:

```bash
/home/ofeai/miniconda3/envs/DreamV3/bin/scope \
  --basedir /home/ofeai/autodl-tmp/logdir \
  --port 8000
```

SSH port forwarding from local machine:

```bash
ssh -L 8000:127.0.0.1:8000 ofeai@SERVER_HOST
```

Then open locally:

```text
http://127.0.0.1:8000
```

## Issues and Fixes

1. System `python` was missing; `python3` was 3.12.3. Fixed by using conda env `DreamV3` with Python 3.11.15.
2. `conda` was not in PATH, but existed at `/home/ofeai/miniconda3/bin/conda`.
3. System `git` was missing. Fixed by downloading the GitHub zip archive.
4. `/root/autodl-tmp` was permission denied for the current user. Used `/home/ofeai/autodl-tmp/logdir` instead.
5. Old `requirements.txt` pins `jax[cuda12]==0.4.33`, unsuitable for RTX 5090/Blackwell. Fixed by installing latest `jax[cuda13]`.
6. `numpy<2` conflicts with JAX 0.10.1. Skipped this old pin.
7. `crafter` was missing from generic requirements. Installed `crafter==1.8.3`.
8. JAX 0.10 no longer accepts old positional `jax.jit()` sharding arguments. Patched DreamerV3 compatibility calls.
9. `debug` config forces CPU. Fixed by adding `--jax.platform cuda`.
10. `python -m scope.viewer` is not available in installed `scope==0.7.0`. Use the `scope` CLI instead.

## Recommended Next Experiments

1. Continue `crafter size50m --run.train_ratio 32` if GPU memory stays below 30 GiB and throughput remains stable.
2. If OOM or large slowdowns occur, switch to:

```bash
/home/ofeai/miniconda3/envs/DreamV3/bin/python dreamerv3/main.py \
  --logdir /home/ofeai/autodl-tmp/logdir/dreamer/crafter_size12m_5090 \
  --configs crafter size12m \
  --run.train_ratio 32 \
  --jax.platform cuda
```

3. If memory pressure persists, reduce batch size first:

```bash
--batch_size 8
```

4. If replay/training pressure is too high, reduce train ratio:

```bash
--run.train_ratio 16
```

5. For quick health checks, reuse `crafter debug --run.train_ratio 32 --jax.platform cuda` before launching long experiments.

## RTX 5090 Notes

- Prefer CUDA 13 JAX wheels on Blackwell.
- Do not downgrade to `jax[cuda12]==0.4.33` unless there is explicit evidence that the installed JAX CUDA 13 stack is broken.
- The size50m config fits on this 32GB RTX 5090 in the current run, using about 25.8GB including desktop display processes.
- A display session is using some VRAM. For maximum headroom, run headless or close GUI applications if possible.
- Keep using explicit `--jax.platform cuda` when combining configs that may override JAX platform.

## 2026-06-15 Update: Logdir Moved Into Current Workspace

The active training logdir has been moved into the current project directory:

```bash
/home/ofeai/DreamV3/logdir/dreamer/crafter_size50m_5090
```

The previous logdir under `/home/ofeai/autodl-tmp/logdir` was copied, not deleted. Training was restarted from the copied checkpoint and is now running from the current-directory logdir.

Active training process:

```text
PID: 794645
Logdir: /home/ofeai/DreamV3/logdir/dreamer/crafter_size50m_5090
```

Current training command:

```bash
cd /home/ofeai/DreamV3
setsid /home/ofeai/miniconda3/envs/DreamV3/bin/python dreamerv3/main.py \
  --logdir /home/ofeai/DreamV3/logdir/dreamer/crafter_size50m_5090 \
  --configs crafter size50m \
  --run.train_ratio 32 \
  --jax.platform cuda \
  >> /home/ofeai/DreamV3/logdir/dreamer/crafter_size50m_5090/train_stdout.log 2>&1 < /dev/null &
```

Realtime Scope viewer is running:

```text
PID: 797206
URL on server: http://127.0.0.1:8000
Basedir: /home/ofeai/DreamV3/logdir
```

If connecting from a local machine via SSH, use:

```bash
ssh -L 8000:127.0.0.1:8000 ofeai@SERVER_HOST
```

Then open:

```text
http://127.0.0.1:8000
```

Monitoring commands:

```bash
tail -f /home/ofeai/DreamV3/logdir/dreamer/crafter_size50m_5090/train_stdout.log
tail -f /home/ofeai/DreamV3/logdir/dreamer/crafter_size50m_5090/metrics.jsonl
nvidia-smi
```
