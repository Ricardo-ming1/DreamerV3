# DreamerV3 Training Artifacts

This directory contains important outputs from the Crafter `size50m` RTX 5090 run.

## GitHub Upload Format

The complete artifact archive is split into chunks below GitHub's 100MB per-file limit:

```text
crafter_size50m_2.77m.tar.zst.part-000
crafter_size50m_2.77m.tar.zst.part-001
...
```

Reconstruct it after cloning:

```bash
cd training_artifacts
cat crafter_size50m_2.77m.tar.zst.part-* > crafter_size50m_2.77m.tar.zst
tar --zstd -xf crafter_size50m_2.77m.tar.zst
```

The extracted directory contains:

- latest complete checkpoint weights: `ckpt/20260616T192326F256097/agent.pkl`
- checkpoint metadata: `step.pkl`, `replay.pkl`, `done`, `ckpt/latest`
- `config.yaml`
- `metrics.jsonl`
- `scores.jsonl`
- `train_stdout.txt`

The full replay buffer is not included because it was about 25GB and caused RAM pressure during training. The checkpoint and scalar logs are the important reusable artifacts.
