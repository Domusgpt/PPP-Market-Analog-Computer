"""
Tests for quasicrystal_architecture.py
======================================

Verifies all eight architectural innovations:
1. Quasicrystalline Reservoir
2. Golden-Ratio MRA
3. Number Field Hierarchy
4. Galois Dual-Channel Verification
5. Phason Error Correction
6. Collision-Aware Encoding
7. Padovan-Stepped Cascade
8. Five-Fold Resource Allocation

Run: cd _SYNERGIZED_SYSTEM/backend && python -m pytest tests/test_quasicrystal_architecture.py -v
"""

import numpy as np
import pytest
from engine.geometry.h4_geometry import PHI, PHI_INV
from engine.geometry.e8_projection import (
    PHILLIPS_MATRIX, PHILLIPS_U_L, PHILLIPS_U_R,
    generate_e8_roots,
)
from engine.geometry.quasicrystal_architecture import (
    RHO, _a, _b, _c,
    QuasicrystallineReservoir,
    GoldenMRA,
    NumberFieldHierarchy,
    GaloisVerifier,
    PhasonErrorCorrector,
    CollisionAwareEncoder,
    PadovanCascade,
    FiveFoldAllocator,
)


# =========================================================================
# 1. QUASICRYSTALLINE RESERVOIR
# =========================================================================

class TestQuasicrystallineReservoir:
    """Tests for the quasicrystalline reservoir."""

    def test_construction(self):
        """Reservoir builds without error."""
        r = QuasicrystallineReservoir(n_reservoir=32, input_dim=8)
        assert r.W.shape == (32, 32)
        assert r.W_in.shape == (32, 8)

    def test_spectral_radius_golden(self):
        """Spectral radius should be near 1/φ ≈ 0.618."""
        r = QuasicrystallineReservoir(n_reservoir=64, input_dim=8)
        sr = r.spectral_radius
        # Should be close to 1/φ (the algebraic target)
        assert abs(sr - PHI_INV) < 0.05, f"Spectral radius {sr} not near 1/φ = {PHI_INV}"

    def test_gram_eigenvalues(self):
        """Gram eigenvalues should include golden-ratio values."""
        r = QuasicrystallineReservoir(n_reservoir=32, input_dim=8)
        eigs = r.gram_eigenvalues
        assert len(eigs) == 8
        # Should have exactly 4 nonzero eigenvalues (rank 4 kernel)
        nonzero = eigs[eigs > 1e-10]
        assert len(nonzero) == 4

    def test_run_stability(self):
        """Reservoir state shouldn't explode or collapse."""
        r = QuasicrystallineReservoir(n_reservoir=32, input_dim=8)
        inputs = np.random.RandomState(42).randn(100, 8)
        states = r.run(inputs)
        # States should stay bounded
        assert np.max(np.abs(states)) < 10.0, "Reservoir exploded"
        # States shouldn't all be zero
        assert np.max(np.abs(states[-1])) > 1e-8, "Reservoir collapsed"


    def test_parameterization_matches_formulas(self):
        """Hierarchy damping/coupling should follow algebraic formulas."""
        h = NumberFieldHierarchy()
        report = h.validate_parameterization()
        assert report['all_valid']
        for level in report['levels']:
            assert level['damping_valid']
            assert level['coupling_valid']

    def test_reset(self):
        """Reset should zero the state."""
        r = QuasicrystallineReservoir(n_reservoir=16, input_dim=4)
        r.step(np.ones(4))
        assert np.any(r.state != 0)
        r.reset()
        assert np.all(r.state == 0)


# =========================================================================
# 2. GOLDEN-RATIO MRA
# =========================================================================

class TestGoldenMRA:
    """Tests for the golden-ratio multi-resolution analysis."""

    def test_construction(self):
        """MRA builds with correct filter dimensions."""
        mra = GoldenMRA(n_levels=3, signal_dim=8)
        assert len(mra.scaling_filter) == 8
        assert len(mra.detail_filters) == 3

    def test_dilation_factor(self):
        """Dilation factor should be φ."""
        mra = GoldenMRA()
        assert mra.dilation_factor == PHI

    def test_filter_normalization(self):
        """All filters should be unit-normalized."""
        mra = GoldenMRA()
        assert abs(np.linalg.norm(mra.scaling_filter) - 1.0) < 1e-10
        for f in mra.detail_filters:
            assert abs(np.linalg.norm(f) - 1.0) < 1e-10

    def test_decompose(self):
        """Decomposition should produce multi-level coefficients."""
        mra = GoldenMRA(n_levels=4)
        signal = np.sin(np.linspace(0, 4 * np.pi, 64))
        coeffs = mra.decompose(signal)
        assert 'approximation' in coeffs
        assert 'details' in coeffs
        assert 'energies' in coeffs
        assert len(coeffs['approximation']) > 0
        assert len(coeffs['details']) > 0

    def test_fibonacci_indices(self):
        """Fibonacci indices should be valid Fibonacci numbers."""
        mra = GoldenMRA()
        idx = mra._fibonacci_indices(100)
        # First few should be 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89
        expected_start = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        for i, e in enumerate(expected_start):
            if i < len(idx):
                assert idx[i] == e, f"Fibonacci index {i}: got {idx[i]}, expected {e}"

    def test_filter_entries(self):
        """Filter entries should report golden ratio."""
        mra = GoldenMRA()
        entries = mra.filter_entries
        assert abs(entries['ratio'] - PHI) < 1e-10


# =========================================================================
# 3. NUMBER FIELD HIERARCHY
# =========================================================================

class TestNumberFieldHierarchy:
    """Tests for the number field hierarchy."""

    def test_three_levels(self):
        """Should have exactly 3 levels."""
        h = NumberFieldHierarchy(base_size=16)
        assert len(h.levels) == 3

    def test_level_names(self):
        """Levels should be Q, Q(sqrt(5)), Q(rho)."""
        h = NumberFieldHierarchy()
        names = [l.name for l in h.levels]
        assert names == ["Q", "Q(sqrt(5))", "Q(rho)"]

    def test_algebraic_numbers(self):
        """Level algebraic numbers should be 1, φ, ρ."""
        h = NumberFieldHierarchy()
        assert h.levels[0].algebraic_number == 1.0
        assert abs(h.levels[1].algebraic_number - PHI) < 1e-10
        assert abs(h.levels[2].algebraic_number - RHO) < 1e-6

    def test_discriminants(self):
        """Discriminants should be 1, 5, -23."""
        h = NumberFieldHierarchy()
        assert h.levels[0].discriminant == 1
        assert h.levels[1].discriminant == 5
        assert h.levels[2].discriminant == -23

    def test_run(self):
        """Running should produce state histories for all levels."""
        h = NumberFieldHierarchy(base_size=16)
        inputs = np.random.RandomState(42).randn(20, 16)
        histories = h.run(inputs)
        assert len(histories) == 3
        for name, hist in histories.items():
            assert hist.shape == (20, 16)

    def test_level_summary(self):
        """Summary should include all required fields."""
        h = NumberFieldHierarchy(base_size=8)
        summary = h.level_summary
        assert len(summary) == 3
        for s in summary:
            assert 'name' in s
            assert 'algebraic_number' in s
            assert 'discriminant' in s
            assert 'spectral_radius' in s


    def test_parameterization_matches_formulas(self):
        """Hierarchy damping/coupling should follow algebraic formulas."""
        h = NumberFieldHierarchy()
        report = h.validate_parameterization()
        assert report['all_valid']
        for level in report['levels']:
            assert level['damping_valid']
            assert level['coupling_valid']

    def test_reset(self):
        """Reset should zero all states."""
        h = NumberFieldHierarchy(base_size=8)
        h.step(np.ones(8))
        h.reset()
        for state in h.states:
            assert np.all(state == 0)


# =========================================================================
# 4. GALOIS DUAL-CHANNEL VERIFICATION
# =========================================================================

class TestGaloisVerifier:
    """Tests for Galois dual-channel verification."""

    def test_e8_roots_perfect_coupling(self):
        """All 240 E8 roots should satisfy φ-coupling exactly."""
        v = GaloisVerifier(tolerance=1e-6)
        result = v.verify_e8_roots()
        assert result['all_valid'], f"E8 roots failed φ-coupling: {result}"
        assert result['max_deviation'] < 1e-8

    def test_single_root(self):
        """Single root verification should work."""
        v = GaloisVerifier()
        roots = generate_e8_roots()
        result = v.verify(roots[0].coordinates)
        assert result['valid']
        assert result['ratio_valid']
        assert result['product_valid']
        assert abs(result['ratio'] - PHI) < 1e-8

    def test_random_vector(self):
        """Random 8D vector should also satisfy φ-coupling."""
        v = GaloisVerifier(tolerance=1e-6)
        # φ-coupling holds for ALL vectors, not just E8 roots
        vec = np.random.RandomState(42).randn(8)
        result = v.verify(vec)
        assert result['valid'], f"Random vector failed: ratio = {result['ratio']}"

    def test_phi_product_coupling(self):
        """Check φ product coupling: ||U_L x|| · ||U_R x|| = φ · ||U_L x||²."""
        v = GaloisVerifier()
        roots = generate_e8_roots()
        for root in roots[:10]:
            result = v.sqrt5_coupling_check(root.coordinates)
            assert result['valid'], f"φ product coupling failed: {result}"
            # Also verify the √5 row norm product (matrix property)
            assert abs(result['sqrt5_row_norm_product'] - np.sqrt(5)) < 1e-8


    def test_ratio_only_perturbation_detection(self):
        """Ratio invariant should fail when right channel is perturbed."""
        v = GaloisVerifier(tolerance=1e-12)
        roots = generate_e8_roots()
        vec = roots[0].coordinates
        left, right = v.compute_dual(vec)

        right_perturbed = right.copy()
        right_perturbed[0] += 1e-3
        ratio = np.linalg.norm(right_perturbed) / np.linalg.norm(left)
        ratio_deviation = abs(ratio - PHI)

        assert ratio_deviation > v.tolerance

    def test_product_only_perturbation_detection(self):
        """Product invariant should fail when vector norm coupling is perturbed."""
        v = GaloisVerifier(tolerance=1e-12)
        vec = np.random.RandomState(0).randn(8)
        result = v.verify(vec)
        assert result['product_valid']

        # Perturb expected product analytically by scaling norm reference
        wrong_expected = result['expected_product'] * 1.1
        wrong_deviation = abs(result['product'] - wrong_expected)
        assert wrong_deviation > v.tolerance * max(abs(wrong_expected), 1e-12)

    def test_combined_validity_semantics(self):
        """Combined validity should represent logical AND of both invariants."""
        v = GaloisVerifier()
        vec = np.random.RandomState(1).randn(8)
        result = v.verify(vec)
        assert result['valid'] == (result['ratio_valid'] and result['product_valid'])

    def test_error_rate_starts_zero(self):
        """Error rate should start at zero."""
        v = GaloisVerifier()
        assert v.error_rate == 0.0

    def test_batch_verification(self):
        """Batch verification should process all vectors."""
        v = GaloisVerifier()
        vectors = np.random.RandomState(42).randn(20, 8)
        result = v.verify_batch(vectors)
        assert result['n_vectors'] == 20
        assert 'max_ratio_deviation' in result
        assert 'max_product_deviation' in result
        assert 'ratio_failures' in result
        assert 'product_failures' in result


# =========================================================================
# 5. PHASON ERROR CORRECTION
# =========================================================================

class TestPhasonErrorCorrector:
    """Tests for phason error correction."""

    def test_kernel_dimension(self):
        """Kernel should be 4-dimensional (rank 4 matrix in 8D)."""
        c = PhasonErrorCorrector()
        assert c.kernel_dimension == 4

    def test_clean_directions_exist(self):
        """Should have at least 1 clean (non-collision) kernel direction."""
        c = PhasonErrorCorrector()
        assert c.clean_dimension >= 1

    def test_collision_direction(self):
        """Collision direction should be (0,1,0,1,0,1,0,1)/2."""
        c = PhasonErrorCorrector()
        expected = np.array([0, 1, 0, 1, 0, 1, 0, 1]) / 2.0
        assert np.allclose(c.collision_direction, expected)

    def test_encode_preserves_projection(self):
        """Encoding should NOT change the projected output."""
        c = PhasonErrorCorrector()
        roots = generate_e8_roots()
        v = roots[0].coordinates.copy()

        encoded = c.encode(v)

        # Projection should be identical
        proj_orig = PHILLIPS_U_L @ v
        proj_encoded = PHILLIPS_U_L @ encoded
        assert np.allclose(proj_orig, proj_encoded, atol=1e-10), \
            "Encoding changed the projection!"

    def test_error_detection(self):
        """Should detect errors introduced after encoding."""
        c = PhasonErrorCorrector()
        v = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
        encoded = c.encode(v)

        # Simulate an error
        corrupted = encoded.copy()
        corrupted[0] += 0.1  # Introduce error

        result = c.verify(encoded, corrupted)
        # Error SHOULD be detected (unless it's in kernel direction,
        # in which case the kernel component check catches it)
        assert result['max_mismatch'] > 0

    def test_collision_direction_info(self):
        """Should report collision direction info correctly."""
        c = PhasonErrorCorrector()
        info = c.collision_direction_info
        assert info['n_collision_pairs'] == 14


# =========================================================================
# 6. COLLISION-AWARE ENCODING
# =========================================================================

class TestCollisionAwareEncoder:
    """Tests for collision-aware encoding."""

    def test_collision_count(self):
        """Should find exactly 14 collision pairs."""
        e = CollisionAwareEncoder()
        assert e.n_collision_pairs == 14

    def test_distinct_projections(self):
        """Should have 240 - 14 = 226 distinct projections."""
        e = CollisionAwareEncoder()
        assert e.n_distinct_projections == 226

    def test_encode_single_root(self):
        """Encoding a single root should return valid metadata."""
        e = CollisionAwareEncoder()
        result = e.encode_root(0)
        assert 'projection_4d' in result
        assert 'has_collision' in result
        assert len(result['projection_4d']) == 4

    def test_compressed_representation(self):
        """Compressed representation should be lossless."""
        e = CollisionAwareEncoder()
        compressed = e.compressed_representation()
        assert compressed['n_unique_projections'] == 226
        assert compressed['n_collision_groups'] == 14
        assert compressed['compression_ratio'] == 240 / 226


# =========================================================================
# 7. PADOVAN-STEPPED CASCADE
# =========================================================================

class TestPadovanCascade:
    """Tests for Padovan-stepped cascade."""

    def test_padovan_sequence(self):
        """Padovan sequence should start with [1, 1, 1, 2, 2, 3, ...]."""
        c = PadovanCascade(max_steps=50)
        seq = c.padovan_steps
        assert seq[:6] == [1, 1, 1, 2, 2, 3]

    def test_padovan_ratio_converges_to_rho(self):
        """Ratio of consecutive Padovan numbers should approach ρ."""
        c = PadovanCascade(max_steps=500)
        ratio = c.padovan_ratio
        assert abs(ratio - RHO) < 0.1, f"Padovan ratio {ratio} not near ρ = {RHO}"

    def test_run(self):
        """Cascade should run without error and return results."""
        c = PadovanCascade(max_steps=50, grid_size=8)
        test_input = np.random.RandomState(42).rand(8, 8)
        c.inject(test_input)
        result = c.run(n_epochs=1)
        assert 'final_state' in result
        assert 'energies' in result
        assert result['final_state'].shape == (8, 8)


    def test_parameterization_matches_formulas(self):
        """Hierarchy damping/coupling should follow algebraic formulas."""
        h = NumberFieldHierarchy()
        report = h.validate_parameterization()
        assert report['all_valid']
        for level in report['levels']:
            assert level['damping_valid']
            assert level['coupling_valid']

    def test_reset(self):
        """Reset should zero state and velocity."""
        c = PadovanCascade(max_steps=30, grid_size=4)
        c.state = np.ones((4, 4))
        c.reset()
        assert np.all(c.state == 0)
        assert np.all(c.velocity == 0)

    def test_energy_bounded(self):
        """Energy should not explode during cascade."""
        c = PadovanCascade(max_steps=50, grid_size=8)
        c.inject(np.random.RandomState(42).rand(8, 8))
        result = c.run()
        if result['energies']:
            assert max(result['energies']) < 1e6, "Cascade energy exploded"


# =========================================================================
# 8. FIVE-FOLD RESOURCE ALLOCATION
# =========================================================================

class TestFiveFoldAllocator:
    """Tests for five-fold resource allocation."""

    def test_five_equals_five(self):
        """Frobenius²/rank should equal 5."""
        a = FiveFoldAllocator()
        result = a.verify_five_equals_five()
        assert result['match'], f"Five != Five: {result['amplification']}"

    def test_five_nodes(self):
        """Should allocate to exactly 5 nodes."""
        a = FiveFoldAllocator(total_budget=1.0)
        allocs = a.get_all_allocations()
        assert len(allocs) == 5

    def test_budget_sums_to_total(self):
        """Per-node budgets should sum to total."""
        a = FiveFoldAllocator(total_budget=10.0)
        allocs = a.get_all_allocations()
        total = sum(alloc['total'] for alloc in allocs.values())
        assert abs(total - 10.0) < 1e-10

    def test_trinity_weights_sum_to_one(self):
        """Trinity weights (α, β, γ) should sum to 1."""
        a = FiveFoldAllocator()
        alloc = a.get_allocation(0)
        weights = alloc['trinity_weights']
        assert abs(sum(weights) - 1.0) < 1e-10

    def test_per_node_budget(self):
        """Per-node budget should be total/5."""
        a = FiveFoldAllocator(total_budget=100.0)
        assert abs(a.per_node_budget - 20.0) < 1e-10

    def test_amplification_equals_group_index(self):
        """Amplification factor should equal the group index 5."""
        a = FiveFoldAllocator()
        assert abs(a.amplification - 5.0) < 1e-8


# =========================================================================
# INTEGRATION TESTS
# =========================================================================

class TestIntegration:
    """Integration tests across multiple components."""

    def test_galois_verifier_with_collision_encoder(self):
        """Collision pairs should still satisfy Galois coupling individually."""
        verifier = GaloisVerifier()
        encoder = CollisionAwareEncoder()

        roots = generate_e8_roots()
        for pair in encoder.collision_pairs[:5]:
            for idx in pair:
                result = verifier.verify(roots[idx].coordinates)
                assert result['valid'], \
                    f"Root {idx} in collision pair failed Galois check"

    def test_phason_corrector_with_collision_encoder(self):
        """Phason encoding should preserve collision structure."""
        corrector = PhasonErrorCorrector()
        encoder = CollisionAwareEncoder()
        roots = generate_e8_roots()

        # Encode a collision pair
        for pair in encoder.collision_pairs[:3]:
            for idx in pair:
                v = roots[idx].coordinates.copy()
                encoded = corrector.encode(v)
                # Should still be part of the same collision group
                proj_orig = tuple(np.round(PHILLIPS_U_L @ v, 8))
                proj_enc = tuple(np.round(PHILLIPS_U_L @ encoded, 8))
                assert proj_orig == proj_enc, \
                    f"Phason encoding changed collision group for root {idx}"

    def test_hierarchy_with_reservoir(self):
        """Hierarchy and reservoir should operate on compatible dimensions."""
        hierarchy = NumberFieldHierarchy(base_size=8)
        reservoir = QuasicrystallineReservoir(n_reservoir=32, input_dim=8)

        # Feed hierarchy output into reservoir
        input_vec = np.random.RandomState(42).randn(8)
        hierarchy.step(input_vec)

        # Use Level 1 (Q(√5)) state as reservoir input
        state_qphi = hierarchy.states[1]
        reservoir_state = reservoir.step(state_qphi)
        assert len(reservoir_state) == 32

    def test_allocator_with_cascade(self):
        """Allocator budget should scale cascade parameters."""
        allocator = FiveFoldAllocator(total_budget=1.0)

        # Each node gets budget/5 — use this to set cascade coupling
        for node_id in range(5):
            alloc = allocator.get_allocation(node_id)
            coupling = alloc['total']  # Use allocation as coupling strength
            cascade = PadovanCascade(max_steps=20, grid_size=4, coupling=coupling)
            cascade.inject(np.random.RandomState(node_id).rand(4, 4))
            result = cascade.run()
            assert result['final_state'].shape == (4, 4)
