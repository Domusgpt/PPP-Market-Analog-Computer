---
name: hemoc-analyze
description: Analyze HEMOC experiment results, compare encoders, or inspect encoding properties. Use when reviewing experimental data, comparing results between experiments, or investigating encoder behavior.
argument-hint: "[results|compare|encoder-check|correlation-matrix]"
---

# HEMOC Analysis Tools

Analyze experimental data based on `$ARGUMENTS`.

## Read First

- `CLAUDE.md` — Known doc contradictions, correct file locations
- HEMOC `docs/PROJECT_STATUS.md` — 15 experiments and their results

## Analysis Modes

### `results` — Summarize All Result Artifacts

Find and parse all JSON files in HEMOC `results/` directory:

```bash
cd /home/user/HEMOC-Stain-Glass-Flower
find results/ -name "*.json" -type f 2>/dev/null
```

For each result file:
1. Read the JSON
2. Extract: experiment name, encoder, decoder, aggregate correlation, per-angle correlations
3. Flag any schema violations

Output as a sorted table (best correlation first):

```
| Experiment | Encoder | Decoder | Avg Corr | Angles Passing | Seed |
|------------|---------|---------|----------|----------------|------|
```

### `compare` — Side-by-Side Experiment Comparison

Compare two or more experiments. Arguments: experiment IDs or file paths.

For each pair:
- Per-angle correlation delta
- Which encoder/decoder combination
- Training data size difference
- Statistical significance (if enough data points)

### `encoder-check` — Inspect Encoder Properties

Check encoder injectivity, sensitivity, and determinism.

Steps:
1. Activate HEMOC venv
2. Import the encoder from `demos/dual_decoder.py`
3. Generate N random angle vectors (default N=100)
4. Encode each → check for collisions (injectivity)
5. Perturb each angle by epsilon → measure output change (sensitivity)
6. Re-encode with same seed → verify identical output (determinism)

```python
import sys
sys.path.insert(0, '/home/user/HEMOC-Stain-Glass-Flower')
from demos.dual_decoder import HybridEncoder
import numpy as np

encoder = HybridEncoder()
np.random.seed(42)
angles = np.random.uniform(0, 2*np.pi, (100, 6))
patterns = [encoder.encode(a) for a in angles]
# Check injectivity: are all patterns unique?
# Check sensitivity: per-angle gradient estimation
```

Report:
```
| Property | Result | Details |
|----------|--------|---------|
| Injectivity | PASS/FAIL | N collisions out of M samples |
| Determinism | PASS/FAIL | Identical outputs for same inputs |
| Angle 0 sensitivity | <value> | Mean output change per radian |
| Angle 1 sensitivity | <value> | ... |
| ... | ... | ... |
```

### `correlation-matrix` — Full Correlation Analysis

If a result file with per-angle data exists, produce:
1. Per-angle bar chart (text-based)
2. Which angles are strongest/weakest
3. Comparison to the 0.916 benchmark (Exp 13)
4. Suggested improvements based on weak angles

## Key Reference Data

From Experiment 13 (audio pipeline, 15K samples, current best):
- Aggregate: 0.916
- Per-angle: approximately [0.94, 0.92, 0.93, 0.88, 0.85, 0.96]
- All 6 angles passing at 0.5 threshold

From Experiment 7 (CNN decoder, first breakthrough):
- Aggregate: 0.657
- 5/6 angles passing
- Angle 4 consistently weakest

## Critical Reminders

- Working encoder: `demos/dual_decoder.py::HybridEncoder` (NOT `main.py::OpticalKirigamiEncoder`)
- Working decoder: `demos/option_e_scaled_cnn.py::AngleRecoveryCNN`
- `src/` directory does NOT exist despite what some docs claim
- PyTorch required for decoder analysis but NOT for encoder-only checks
