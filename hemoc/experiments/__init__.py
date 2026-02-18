"""
hemoc.experiments -- Experiment Registry and Runners
=====================================================

The canonical experiment registry binds every experiment to its code,
config, renderer version, and success criteria.  Each experiment is
tagged by renderer dependence:

  - discrete_invariant:     Proxy-safe (discrete state tasks)
  - continuous_manifold:    Proxy-invalid (continuous regression)
  - resolution_dependent:   Proxy-invalid (foveation/zoom tasks)
  - noise_dependent:        Proxy-risk (noise robustness comparisons)

Modules
-------
registry     Load and query the experiments.yaml registry
"""

from hemoc.experiments.registry import ExperimentRegistry, Experiment

__all__ = ["ExperimentRegistry", "Experiment"]
