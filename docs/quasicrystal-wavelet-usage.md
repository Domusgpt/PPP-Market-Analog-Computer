# Quasicrystal Wavelet Usage Guide

## Purpose

This note documents how to select the wavelet backend in the unified backend feature extractor after Phase B/C integration.

## Configuration

Use `FeatureConfig.wavelet_family` to choose decomposition architecture:

- `dyadic` (default): classical 2-adic downsampling via `WaveletDecomposer`.
- `golden`: φ-adic/Fibonacci downsampling via `GoldenMRAAdapter` backed by `GoldenMRA`.

Optional `wavelet_type` still applies for dyadic mode (`haar`, `db2`, ...), while golden mode ignores dyadic filter families and uses the Phillips-derived golden bank.

```python
from engine.features import FeatureExtractor, FeatureConfig, FeatureType

cfg = FeatureConfig(
    features={FeatureType.WAVELET},
    wavelet_family="golden",  # or "dyadic"
    wavelet_levels=3,
    normalize=True,
)

features = FeatureExtractor(cfg).extract(image)
print(features.feature_names[:5])
```

## Naming Convention

Wavelet feature names are prefixed by backend family:

- Dyadic: `wavelet_dyadic_*`
- Golden: `wavelet_golden_*`

This makes downstream telemetry and ML pipelines backend-explicit and prevents accidental feature mixing.

## Selection Guidance

- Choose **dyadic** for compatibility with conventional signal processing baselines.
- Choose **golden** for quasicrystal-native, φ-symmetric multi-resolution analysis aligned with the Phillips projection framework.

## Validation

Run:

```bash
cd _SYNERGIZED_SYSTEM/backend
pytest -q tests/test_features.py tests/test_quasicrystal_architecture.py
```

The tests verify deterministic dimensions, stable naming, and compatibility across both families.
