---
name: hemoc-experiment
description: Set up, run, or analyze a HEMOC experiment with proper schema compliance, deterministic seeds, and result validation. Use when running encode-decode experiments, training runs, or benchmark comparisons.
argument-hint: "[encode-test|roundtrip|benchmark|training]"
disable-model-invocation: true
---

# HEMOC Experiment Runner

Set up and execute experiments following quality standards from CLAUDE.md:
- Deterministic seeds (recorded in metadata)
- Schema-compliant result output
- Auto-validation against benchmark contract
- Standardized output format

## Pre-Flight Checklist

Before ANY experiment:

```bash
cd /home/user/HEMOC-Stain-Glass-Flower

# 1. Venv active with deps?
source .venv/bin/activate
python -c "import numpy, scipy; print('Core deps OK')"

# 2. PyTorch needed? (for decoder/training experiments)
python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "torch NOT installed — encoder-only experiments still work"

# 3. Working pipeline files present?
test -f demos/dual_decoder.py && echo "HybridEncoder: OK" || echo "MISSING: demos/dual_decoder.py"
test -f demos/option_e_scaled_cnn.py && echo "AngleRecoveryCNN: OK" || echo "MISSING: demos/option_e_scaled_cnn.py"

# 4. Results directory?
mkdir -p results

# 5. Last experiment number?
echo "Existing results: $(ls results/*.json 2>/dev/null | wc -l) files"
```

Read `docs/PROJECT_STATUS.md` for last experiment number. Continue from Exp 16+.

## Context Files

1. PPP `CLAUDE.md` — Quality standards, known pitfalls
2. HEMOC `docs/PROJECT_STATUS.md` — 15 existing experiments
3. HEMOC `FINAL_ANALYSIS_REPORT.md` — V4 Plastic Encoder benchmark
4. HEMOC `docs/benchmark_contract.md` (on `hemoc-visual-system-init-*`) — Result schema

## Experiment Types

### `encode-test` — Encoder Injectivity & Sensitivity
- **Injectivity**: 0 collisions across N angle vectors
- **Sensitivity**: each of 6 angles measurably affects output
- **Determinism**: same seed + angles = identical output
- Does NOT require PyTorch. Uses `demos/dual_decoder.py::HybridEncoder`.

### `roundtrip` — Encode-Decode Correlation
- Full HybridEncoder → AngleRecoveryCNN cycle
- REQUIRES PyTorch for decoder
- Compare to Exp 13 baseline (0.916)

### `benchmark` — Encoder Comparison
Head-to-head encoder comparison. Reference from FINAL_ANALYSIS_REPORT:

| Encoder | Cosine Sim | Rank |
|---------|-----------|------|
| V4 Plastic | 0.6261 | 1 |
| V3.1 600-cell | 0.4558 | 2 |
| Hybrid | 0.4168 | 3 |
| Trinity | 0.0774 | 4 |

### `training` — CNN Training Run
REQUIRES PyTorch. Parameters: dataset_size (default 15000), epochs, batch_size, encoder_type.

## Demo Script Reference

| Category | Script | Description |
|----------|--------|-------------|
| Core encoder | `demos/dual_decoder.py` | HybridEncoder class |
| Core decoder | `demos/option_e_scaled_cnn.py` | AngleRecoveryCNN + training |
| Untested | `demos/domain_diverse_training.py` | Multi-domain (NEVER RUN) |
| Benchmark | `demos/benchmark_suite.py` | Performance testing |
| Runner | `demos/run_all_demos.py` | Executes all demo_*.py |
| Basic | `demos/demo_basic_encoding.py` | Basic encode demo |
| Dynamics | `demos/demo_cascade_dynamics.py` | Cascade visualization |
| Angles | `demos/demo_angle_comparison.py` | Angle comparison |
| Streaming | `demos/demo_streaming.py` | Streaming encoder |
| Quaternion | `demos/demo_quaternion_moire.py` | Quaternion patterns |
| Market | `demos/demo_interactive_market.py` | Market data encoding |
| Anomaly | `demos/demo_anomaly.py` | Anomaly detection |
| Live | `demos/demo_live_data.py` | Live data encoding |

## Result Schema

Every experiment MUST produce `results/exp_YYYYMMDD_HHMMSS_<name>.json`:

```json
{
  "experiment": {
    "id": "exp_<timestamp>_<name>",
    "name": "<descriptive name>",
    "number": 16,
    "type": "<encode-test|roundtrip|benchmark|training>",
    "seed": 42,
    "timestamp": "<ISO 8601>"
  },
  "config": {
    "encoder": "<class name>",
    "decoder": "<class name or null>",
    "dataset_size": null,
    "resolution": [64, 64],
    "angles": 6
  },
  "results": {
    "per_angle_correlation": [],
    "aggregate_correlation": null,
    "passing_angles": null,
    "threshold": 0.5
  },
  "environment": {
    "python": "<version>",
    "torch": "<version or 'not installed'>",
    "numpy": "<version>",
    "platform": "<os>",
    "gpu": "<gpu name or 'cpu'>"
  }
}
```

## Execution Steps

1. Run pre-flight checklist
2. Set seed: `PYTHONHASHSEED=42` + `np.random.seed(42)` + `torch.manual_seed(42)`
3. Run experiment using appropriate `demos/` script
4. Save result JSON to `results/`
5. Validate: `python scripts/validate_results_schema.py` (if available)
6. Report summary table with per-angle correlations vs baselines

## Critical Warnings

- `src/` does NOT exist. Working code is in `demos/`.
- `main.py::OpticalKirigamiEncoder` is the OLD encoder. Use `demos/dual_decoder.py::HybridEncoder`.
- Cross-domain zero-shot transfer FAILS (-0.027). Calibration (100 samples) improves to 0.22.
- Always record GPU vs CPU — decoder results can differ.
