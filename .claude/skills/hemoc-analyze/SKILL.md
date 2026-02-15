---
name: hemoc-analyze
description: Analyze HEMOC experiment results, compare encoders, or inspect encoding properties. Use when reviewing experimental data, comparing results between experiments, or investigating encoder behavior.
argument-hint: "[results|compare|encoder-check|correlation-matrix|history|gaps]"
---

# HEMOC Analysis Tools

Analyze experimental data based on `$ARGUMENTS`.

## Read First

- PPP `CLAUDE.md` — Known contradictions, correct file paths, research findings
- HEMOC `docs/PROJECT_STATUS.md` — 15 experiments and results
- HEMOC `FINAL_ANALYSIS_REPORT.md` — V4 Plastic Encoder benchmark

## Analysis Modes

### `results` — Summarize All Result Artifacts

Find and parse all JSON files in HEMOC `results/`:
```bash
cd /home/user/HEMOC-Stain-Glass-Flower
find results/ -name "*.json" -type f 2>/dev/null
```

For each: extract name, encoder, decoder, correlation, per-angle data, seed.
Output sorted by correlation (best first):
```
| Experiment | Encoder | Decoder | Avg Corr | Angles Passing | Seed |
|------------|---------|---------|----------|----------------|------|
```

### `compare` — Side-by-Side Experiment Comparison

Compare two+ experiments by ID or file path:
- Per-angle correlation delta
- Encoder/decoder differences
- Training data size differences
- Configuration differences

### `encoder-check` — Inspect Encoder Properties

Check injectivity, sensitivity, determinism of HybridEncoder:

```python
import sys
sys.path.insert(0, '/home/user/HEMOC-Stain-Glass-Flower')
from demos.dual_decoder import HybridEncoder
import numpy as np

encoder = HybridEncoder()
np.random.seed(42)
angles = np.random.uniform(0, 2*np.pi, (100, 6))
patterns = [encoder.encode(a) for a in angles]
```

Report:
```
| Property | Result | Details |
|----------|--------|---------|
| Injectivity | PASS/FAIL | N collisions / M samples |
| Determinism | PASS/FAIL | Same input → same output |
| Angle 0-5 sensitivity | <values> | Mean change per radian |
```

### `correlation-matrix` — Full Correlation Analysis

For a result file with per-angle data:
1. ASCII bar chart per angle
2. Strongest/weakest angle identification
3. Comparison to Exp 13 baseline (0.916)
4. Improvement suggestions for weak angles

### `history` — Full 15-Experiment Progression

Trace the research arc from PROJECT_STATUS.md:

| Exp | What | Correlation | Pass | Key Lesson |
|-----|------|------------|------|------------|
| 1-4 | MLP decoders | 0.001-0.15 | 0/6 | MLPs can't learn from moire |
| 5 | Dual decoder | 0.405 | 3/6 | Hybrid encoder helps |
| 7 | CNN decoder | 0.657 | 5/6 | **CNN breakthrough** |
| 8 | Pure V3 + CNN | 0.090 | 0/6 | Hypercube-only fails |
| 9-11 | ViT attempts | 0.14-0.35 | — | ViT underperforms CNN |
| 12 | Audio 8K | 0.894 | 6/6 | Audio maps naturally |
| 13 | Scaled CNN 15K | **0.916** | 6/6 | More data helps |
| 14-15 | Cross-domain | -0.027 | 0/6 | Zero-shot fails |

Key inflection points:
- **Exp 7**: CNN proved spatial processing works on moire
- **Exp 12**: Audio domain natural fit discovered
- **Exp 13**: Peak result (0.916) — current benchmark
- **Exp 14-15**: Cross-domain failure → Phase C motivation

### `gaps` — Missing Experiments

Experiments from Recovery Plan that have NOT been run:

| Experiment | Phase | Status | Blocker |
|-----------|-------|--------|---------|
| Domain-diverse training | C | NEVER RUN | GPU required |
| Direct-feature MLP baseline | D | NEVER RUN | — |
| Random pattern control | D | NEVER RUN | — |
| Simple visual baseline | D | NEVER RUN | — |
| Hypercube on/off ablation | D | NEVER RUN | — |
| Golden-ratio vs alternatives | D | NEVER RUN | — |
| Resolution sweep (32/64/128) | D | NEVER RUN | — |

## Encoder Benchmark Reference

From FINAL_ANALYSIS_REPORT.md:

| Encoder | Cosine Sim | Rank |
|---------|-----------|------|
| V4 Plastic | 0.6261 | 1 |
| V3.1 600-cell | 0.4558 | 2 |
| Hybrid | 0.4168 | 3 |
| Trinity | 0.0774 | 4 |

**Why V4 won**: Plastic Ratio (rho ~ 1.32, Pisot number) scaling creates frequency
components that remain distinct across scales. CNN reads as depth vs orientation.

## Key Reference Data

**Exp 13** (best): 0.916 avg, 6/6 passing, 15K audio samples
**Exp 7** (breakthrough): 0.657 avg, 5/6 passing, CNN first success
**Angle 4**: Historically weakest across experiments

## Critical Reminders

- Working encoder: `demos/dual_decoder.py::HybridEncoder`
- Working decoder: `demos/option_e_scaled_cnn.py::AngleRecoveryCNN`
- `src/` does NOT exist despite doc claims
- PyTorch needed for decoder, NOT for encoder-only checks
- Centroid-zero: tesseract vertices sum to origin → MLPs fail → CNNs succeed
