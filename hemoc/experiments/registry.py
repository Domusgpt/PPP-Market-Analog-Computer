"""
Experiment Registry
====================

Loads the canonical experiments.yaml and provides typed access to
experiment definitions.  Each experiment has:

  - id:                Unique identifier (e.g., "exp-13")
  - name:              Human-readable name
  - track:             "A" (feature decoding) or "B" (Phillips/cognitive)
  - renderer_dependence: One of discrete_invariant, continuous_manifold,
                        resolution_dependent, noise_dependent
  - code_path:         Path to the runner script/module
  - success_criteria:  Metric thresholds for pass/fail
  - status:            proven / untested / invalidated / re-verified
  - result_artifact:   Path to stored result JSON/MD (if any)
  - notes:             Free-form context
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import yaml


REGISTRY_PATH = Path(__file__).parent / "experiments.yaml"


@dataclass
class Experiment:
    """A single experiment definition from the registry."""
    id: str
    name: str
    track: str
    renderer_dependence: str
    code_path: str
    success_criteria: Dict
    status: str
    result_artifact: Optional[str] = None
    notes: Optional[str] = None
    depends_on: Optional[List[str]] = field(default_factory=list)


class ExperimentRegistry:
    """
    Load and query the canonical experiment registry.

    Usage
    -----
        reg = ExperimentRegistry()
        exp = reg.get("exp-13")
        proxy_safe = reg.filter_by_renderer_dependence("discrete_invariant")
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or REGISTRY_PATH
        self._experiments: Dict[str, Experiment] = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        with open(self.path, 'r') as f:
            data = yaml.safe_load(f)
        if not data or "experiments" not in data:
            return
        for entry in data["experiments"]:
            exp = Experiment(
                id=entry["id"],
                name=entry["name"],
                track=entry.get("track", "A"),
                renderer_dependence=entry.get("renderer_dependence", "unknown"),
                code_path=entry.get("code_path", ""),
                success_criteria=entry.get("success_criteria", {}),
                status=entry.get("status", "untested"),
                result_artifact=entry.get("result_artifact"),
                notes=entry.get("notes"),
                depends_on=entry.get("depends_on", []),
            )
            self._experiments[exp.id] = exp

    def get(self, exp_id: str) -> Optional[Experiment]:
        return self._experiments.get(exp_id)

    def list_all(self) -> List[Experiment]:
        return list(self._experiments.values())

    def filter_by_track(self, track: str) -> List[Experiment]:
        return [e for e in self._experiments.values() if e.track == track]

    def filter_by_status(self, status: str) -> List[Experiment]:
        return [e for e in self._experiments.values() if e.status == status]

    def filter_by_renderer_dependence(self, dep: str) -> List[Experiment]:
        return [e for e in self._experiments.values()
                if e.renderer_dependence == dep]

    def proxy_safe_experiments(self) -> List[Experiment]:
        return self.filter_by_renderer_dependence("discrete_invariant")

    def proxy_invalid_experiments(self) -> List[Experiment]:
        return [e for e in self._experiments.values()
                if e.renderer_dependence in ("continuous_manifold",
                                              "resolution_dependent")]

    def summary(self) -> Dict:
        by_status = {}
        by_track = {}
        by_renderer = {}
        for e in self._experiments.values():
            by_status[e.status] = by_status.get(e.status, 0) + 1
            by_track[e.track] = by_track.get(e.track, 0) + 1
            by_renderer[e.renderer_dependence] = by_renderer.get(
                e.renderer_dependence, 0) + 1
        return {
            "total_experiments": len(self._experiments),
            "by_status": by_status,
            "by_track": by_track,
            "by_renderer_dependence": by_renderer,
        }
