# DreamerV3 RTX 5090 Crafter Run

This fork contains a working DreamerV3 setup for a single NVIDIA GeForce RTX 5090 and a trained Crafter `size50m` checkpoint artifact.

## Training Result

| Item | Value |
| --- | --- |
| Task | `crafter_reward` |
| Config | `crafter size50m` |
| Hardware | Single NVIDIA GeForce RTX 5090, 32GB VRAM |
| Driver / CUDA | NVIDIA driver 580.159.03, CUDA 13.0 |
| Python | 3.11.15 |
| JAX | 0.10.1 with `jax[cuda13]` |
| Training reached | About 2.77M environment steps |
| Recent 100 episode score mean | 12.15 |
| Recent 100 episode score max | 15.10 |
| Earlier 1.1M score mean | 8.34 |

The run improved from a recent-100 score mean of about `8.34` at 1.1M steps to `12.15` at about 2.77M steps. Training toward 5M was stopped early by system RAM pressure from the replay buffer, not by GPU memory exhaustion.

## Included Training Artifact

The important training outputs are stored under:

```text
training_artifacts/
```

The full artifact is split into files below GitHub's 100MB hard file limit:

```text
training_artifacts/crafter_size50m_2.77m.tar.zst.part-000
training_artifacts/crafter_size50m_2.77m.tar.zst.part-001
training_artifacts/crafter_size50m_2.77m.tar.zst.part-002
training_artifacts/crafter_size50m_2.77m.tar.zst.part-003
training_artifacts/crafter_size50m_2.77m.tar.zst.part-004
```

Reconstruct and extract after cloning:

```bash
cd training_artifacts
cat crafter_size50m_2.77m.tar.zst.part-* > crafter_size50m_2.77m.tar.zst
tar --zstd -xf crafter_size50m_2.77m.tar.zst
```

The extracted directory contains:

```text
crafter_size50m_2.77m/
  ckpt/20260616T192326F256097/agent.pkl
  ckpt/20260616T192326F256097/step.pkl
  ckpt/20260616T192326F256097/replay.pkl
  ckpt/20260616T192326F256097/done
  ckpt/latest
  config.yaml
  metrics.jsonl
  scores.jsonl
  train_stdout.txt
```

The 25GB replay buffer is intentionally not included. The checkpoint weights, config, metrics, scores, and stdout log are the reusable training results.

## Restore Checkpoint Layout

DreamerV3 expects checkpoints under a logdir `ckpt/` directory. One simple restore layout is:

```bash
mkdir -p logdir/dreamer/crafter_size50m_5090/ckpt
cp -r training_artifacts/crafter_size50m_2.77m/ckpt/20260616T192326F256097 \
  logdir/dreamer/crafter_size50m_5090/ckpt/
printf '20260616T192326F256097' > logdir/dreamer/crafter_size50m_5090/ckpt/latest
```

## RTX 5090 Setup Notes

The upstream `requirements.txt` pins an older CUDA 12 JAX build. For RTX 5090 / Blackwell, use the CUDA 13 JAX wheel instead:

```bash
conda create -y -n DreamV3 python=3.11 pip
conda activate DreamV3
pip install -U "jax[cuda13]"
```

Install the remaining requirements while skipping the old JAX/CUDA12 pins and `numpy<2`, because JAX 0.10 requires NumPy 2:

```bash
awk '!/^jax\[cuda12\]==/ && !/^nvidia-cuda-nvcc-cu12/ && !/^numpy<2/' \
  requirements.txt > /tmp/dreamerv3_requirements_filtered.txt
pip install -r /tmp/dreamerv3_requirements_filtered.txt
pip install crafter
```

This repo also includes small source compatibility patches for JAX 0.10, changing old positional `jax.jit()` sharding arguments to keyword arguments.

## Run Commands

Debug run:

```bash
python dreamerv3/main.py \
  --logdir /home/ofeai/DreamerV3/logdir/dreamer/debug_crafter \
  --configs crafter debug \
  --run.train_ratio 32 \
  --jax.platform cuda
```

Crafter `size50m` run:

```bash
python dreamerv3/main.py \
  --logdir /home/ofeai/DreamerV3/logdir/dreamer/crafter_size50m_5090 \
  --configs crafter size50m \
  --run.train_ratio 32 \
  --jax.platform cuda
```

Continue toward 5M steps with reduced replay size to avoid RAM exhaustion:

```bash
python dreamerv3/main.py \
  --logdir /home/ofeai/DreamerV3/logdir/dreamer/crafter_size50m_5090 \
  --configs crafter size50m \
  --run.train_ratio 32 \
  --run.steps 5e6 \
  --replay.size 2e6 \
  --jax.platform cuda
```

## View Curves

This environment used the `scope` CLI entry point:

```bash
scope --basedir /home/ofeai/DreamerV3/logdir --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

Important curves:

```text
episode/score
episode/length
replay/replay_ratio
train/loss/image
train/loss/dyn
train/loss/rew
train/rand/action
fps/policy
fps/train
usage/nvsmi/compute_avg/gpu0
```

For detailed environment notes, see:

```text
dreamerv3_training_notes.md
```

---

# Mastering Diverse Domains through World Models

A reimplementation of [DreamerV3][paper], a scalable and general reinforcement
learning algorithm that masters a wide range of applications with fixed
hyperparameters.

![DreamerV3 Tasks](https://user-images.githubusercontent.com/2111293/217647148-cbc522e2-61ad-4553-8e14-1ecdc8d9438b.gif)

If you find this code useful, please reference in your paper:

```
@article{hafner2025dreamerv3,
  title={Mastering diverse control tasks through world models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={Nature},
  pages={1--7},
  year={2025},
  publisher={Nature Publishing Group}
}
```

To learn more:

- [Research paper][paper]
- [Project website][website]
- [Twitter summary][tweet]

## DreamerV3

DreamerV3 learns a world model from experiences and uses it to train an actor
critic policy from imagined trajectories. The world model encodes sensory
inputs into categorical representations and predicts future representations and
rewards given actions.

![DreamerV3 Method Diagram](https://user-images.githubusercontent.com/2111293/217355673-4abc0ce5-1a4b-4366-a08d-64754289d659.png)

DreamerV3 masters a wide range of domains with a fixed set of hyperparameters,
outperforming specialized methods. Removing the need for tuning reduces the
amount of expert knowledge and computational resources needed to apply
reinforcement learning.

![DreamerV3 Benchmark Scores](https://github.com/danijar/dreamerv3/assets/2111293/0fe8f1cf-6970-41ea-9efc-e2e2477e7861)

Due to its robustness, DreamerV3 shows favorable scaling properties. Notably,
using larger models consistently increases not only its final performance but
also its data-efficiency. Increasing the number of gradient steps further
increases data efficiency.

![DreamerV3 Scaling Behavior](https://user-images.githubusercontent.com/2111293/217356063-0cf06b17-89f0-4d5f-85a9-b583438c98dd.png)

# Instructions

The code has been tested on Linux and Mac and requires Python 3.11+.

## Docker

You can either use the provided `Dockerfile` that contains instructions or
follow the manual instructions below.

## Manual

Install [JAX][jax] and then the other dependencies:

```sh
pip install -U -r requirements.txt
```

Training script:

```sh
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/{timestamp} \
  --configs crafter \
  --run.train_ratio 32
```

To reproduce results, train on the desired task using the corresponding config,
such as `--configs atari --task atari_pong`.

View results:

```sh
pip install -U scope
python -m scope.viewer --basedir ~/logdir --port 8000
```

Scalar metrics are also writting as JSONL files.

# Tips

- All config options are listed in `dreamerv3/configs.yaml` and you can
  override them as flags from the command line.
- The `debug` config block reduces the network size, batch size, duration
  between logs, and so on for fast debugging (but does not learn a good model).
- By default, the code tries to run on GPU. You can switch to CPU or TPU using
  the `--jax.platform cpu` flag.
- You can use multiple config blocks that will override defaults in the
  order they are specified, for example `--configs crafter size50m`.
- By default, metrics are printed to the terminal, appended to a JSON lines
  file, and written as Scope summaries. Other outputs like WandB and
  TensorBoard can be enabled in the training script.
- If you get a `Too many leaves for PyTreeDef` error, it means you're
  reloading a checkpoint that is not compatible with the current config. This
  often happens when reusing an old logdir by accident.
- If you are getting CUDA errors, scroll up because the cause is often just an
  error that happened earlier, such as out of memory or incompatible JAX and
  CUDA versions. Try `--batch_size 1` to rule out an out of memory error.
- Many environments are included, some of which require installing additional
  packages. See the `Dockerfile` for reference.
- To continue stopped training runs, simply run the same command line again and
  make sure that the `--logdir` points to the same directory.

# Disclaimer

This repository contains a reimplementation of DreamerV3 based on the open
source DreamerV2 code base. It is unrelated to Google or DeepMind. The
implementation has been tested to reproduce the official results on a range of
environments.

[jax]: https://github.com/google/jax#pip-installation-gpu-cuda
[paper]: https://arxiv.org/pdf/2301.04104
[website]: https://danijar.com/dreamerv3
[tweet]: https://twitter.com/danijarh/status/1613161946223677441
