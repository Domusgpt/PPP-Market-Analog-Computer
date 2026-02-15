---
name: hemoc-experiment
description: Set up, run, or analyze a HEMOC experiment with proper schema compliance, deterministic seeds, and result validation. Use when running encode-decode experiments, training runs, or benchmark comparisons.
argument-hint: "[experiment-name] [optional: encoder-type]"
disable-model-invocation: true
---

# HEMOC Experiment Runner

Set up and execute an experiment following the project's quality standards:
- Deterministic seeds (recorded in metadata)
- Schema-compliant result output
- Auto-validation against benchmark contract
- Standardized output format

## Context Files to Read First

Before running any experiment, read these for the current state:
1. `CLAUDE.md` — Quality standards and known pitfalls
2. HEMOC `docs/PROJECT_STATUS.md` — Existing experiment numbering (continue from Exp 16+)
3. HEMOC `docs/benchmark_contract.md` (on `hemoc-visual-system-init-*` branch) — Result schema

## Experiment Types

Based on `$ARGUMENTS`, determine the experiment type:

### Type: `encode-test` — Encoder Injectivity / Sensitivity
Run the encoder on N random angle vectors. Verify:
- Injectivity: no two angle vectors produce identical patterns
- Sensitivity: each of 6 angles measurably affects the output
- Determinism: same seed + angles = identical output

### Type: `roundtrip` — Encode-Decode Correlation
Run full encode (HybridEncoder) → decode (AngleRecoveryCNN) cycle.
Report per-angle correlation and aggregate.

### Type: `benchmark` — Encoder Comparison
Compare encoders head-to-head:
- HybridEncoder (baseline)
- V4 Plastic Encoder (current best)
- Any new encoder variant

### Type: `training` — CNN Training Run
Train AngleRecoveryCNN on generated data.
Parameters to set: dataset_size, epochs, batch_size, encoder_type.

## Result Schema

Every experiment MUST produce a JSON result file at `results/exp_YYYYMMDD_HHMMSS_<name>.json`:

```json
{
  "experiment": {
    "id": "exp_<timestamp>_<name>",
    "name": "<descriptive name>",
    "number": <next in sequence>,
    "seed": 42,
    "timestamp": "<ISO 8601>"
  },
  "config": {
    "encoder": "<encoder class name>",
    "decoder": "<decoder class name>",
    "dataset_size": <N>,
    "resolution": [64, 64],
    "angles": 6
  },
  "results": {
    "per_angle_correlation": [<6 floats>],
    "aggregate_correlation": <float>,
    "passing_angles": <int 0-6>,
    "threshold": 0.5
  },
  "environment": {
    "python": "<version>",
    "torch": "<version or 'not installed'>",
    "platform": "<os>",
    "gpu": "<gpu name or 'cpu'>"
  }
}
```

## Execution Steps

1. **Activate HEMOC venv**: `source ../HEMOC-Stain-Glass-Flower/.venv/bin/activate`
2. **Set seed**: `PYTHONHASHSEED=42` + numpy/torch seeds in code
3. **Run experiment**: Use appropriate demo script from `demos/`
4. **Save result**: Write JSON to `results/` with the schema above
5. **Validate**: Run `python scripts/validate_results_schema.py` if available
6. **Report**: Print summary table with per-angle correlations

## Key Files

| File | Contains |
|------|----------|
| `demos/dual_decoder.py` | HybridEncoder class |
| `demos/option_e_scaled_cnn.py` | AngleRecoveryCNN class, training loop |
| `demos/domain_diverse_training.py` | Multi-domain training (NEVER RUN) |
| `results/` | Existing result artifacts |
| `main.py` | OLD OpticalKirigamiEncoder (not the working pipeline) |

## Critical Warnings

- PyTorch may not be installed. Check with `python -c "import torch; print(torch.__version__)"` first.
- The `src/` directory does NOT exist. All working code is in `demos/`.
- Cross-domain zero-shot transfer FAILS. Don't expect it to work without domain-specific training.
- Always record whether GPU or CPU was used — results differ.
