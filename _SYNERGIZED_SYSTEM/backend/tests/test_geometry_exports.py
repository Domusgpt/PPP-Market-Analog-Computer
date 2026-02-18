"""Package-level geometry export smoke tests."""

from engine.geometry import (
    RHO,
    QuasicrystallineReservoir,
    GoldenMRA,
    NumberFieldHierarchy,
    GaloisVerifier,
    PhasonErrorCorrector,
    CollisionAwareEncoder,
    PadovanCascade,
    FiveFoldAllocator,
)


def test_quasicrystal_exports_resolve():
    assert RHO > 1.0
    assert QuasicrystallineReservoir is not None
    assert GoldenMRA is not None
    assert NumberFieldHierarchy is not None
    assert GaloisVerifier is not None
    assert PhasonErrorCorrector is not None
    assert CollisionAwareEncoder is not None
    assert PadovanCascade is not None
    assert FiveFoldAllocator is not None
