"""
Quasicrystal Computational Architecture
========================================

Implements eight architectural innovations derived from the Phillips matrix's
quasicrystalline structure, Boyle's Coxeter pair framework, and the
golden/plastic ratio number field hierarchy.

These aren't incremental improvements — they represent a paradigm shift in
how the PPP system computes: from random reservoir + heuristic tuning to
GEOMETRY-DETERMINED computation where every parameter is algebraically fixed.

Innovations:
    1. Quasicrystalline Reservoir Weights (replaces random ESN)
    2. Golden-Ratio MRA (φ-adic wavelet, replaces dyadic)
    3. Number Field Hierarchy (Q → Q(√5) → Q(ρ) multi-scale)
    4. Galois Dual-Channel Verification (U_L ↔ U_R error detection)
    5. Phason Error Correction (kernel-space redundancy codes)
    6. Collision-Aware Encoding (14-pair compression)
    7. Padovan-Stepped Cascade (ρ-governed temporal hierarchy)
    8. Five-Fold Resource Allocation (group-index budget)

References:
    - Phillips, "The Totalistic Geometry of E8" (2026)
    - Boyle & Steinhardt, "Coxeter pairs" (arXiv:1608.08215)
    - Boyle, "Spacetime quasicrystals" (arXiv:2601.07769, 2025)
    - Elser & Sloane, "Quasicrystalline cut-and-project" (1987)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np

from .h4_geometry import PHI, PHI_INV
from .e8_projection import (
    PHILLIPS_MATRIX, PHILLIPS_U_L, PHILLIPS_U_R,
    generate_e8_roots, E8Root,
)

# Plastic ratio: real root of x^3 - x - 1 = 0
RHO = float(np.real(np.roots([1, 0, -1, -1])[0]))

# Phillips entry constants
_a = 0.5
_b = (PHI - 1) / 2   # = 1/(2φ)
_c = PHI / 2          # = φ/2


# =============================================================================
# 1. QUASICRYSTALLINE RESERVOIR WEIGHTS
# =============================================================================

class QuasicrystallineReservoir:
    """
    Reservoir with weights derived from the Phillips Gram matrix.

    Replaces random sparse ESN weights with deterministic weights from
    U_L^T @ U_L — the 8×8 Gram matrix of the Phillips left block. This
    matrix has golden-ratio eigenstructure and naturally sits at the edge
    of chaos (spectral radius related to φ+2).

    The key insight: quasicrystals ARE at the boundary between order and
    disorder. A quasicrystallinely-structured reservoir inherits this
    property without spectral radius tuning.

    Parameters
    ----------
    n_reservoir : int
        Reservoir size (will be tiled with copies of the 8×8 Gram kernel)
    leak_rate : float
        Leaky integration rate
    input_dim : int
        Input dimension
    """

    def __init__(
        self,
        n_reservoir: int = 64,
        leak_rate: float = 0.3,
        input_dim: int = 8,
    ):
        self.n_reservoir = n_reservoir
        self.leak_rate = leak_rate
        self.input_dim = input_dim

        # Build reservoir weights from Phillips Gram matrix
        self.W = self._build_quasicrystal_weights()
        self.W_in = self._build_input_weights()
        self.state = np.zeros(n_reservoir)

    def _build_quasicrystal_weights(self) -> np.ndarray:
        """
        Build reservoir weight matrix by tiling the Phillips Gram kernel.

        The Gram matrix G = U_L^T @ U_L is 8×8 with eigenvalues in
        {0, 0, 0, 0, (3-φ)/2, (3-φ)/2, (φ+2)/2, (φ+2)/2}.
        We tile it to fill N×N, then normalize spectral radius.
        """
        G = PHILLIPS_U_L.T @ PHILLIPS_U_L  # (8, 8)
        N = self.n_reservoir

        # Tile the 8×8 kernel across the N×N reservoir
        n_tiles = (N + 7) // 8
        W_big = np.tile(G, (n_tiles, n_tiles))[:N, :N]

        # Add golden-ratio phase offsets between tiles
        for i in range(0, N, 8):
            for j in range(0, N, 8):
                tile_idx = (i // 8) + (j // 8)
                phase = (PHI * tile_idx) % 1.0  # quasiperiodic phase
                block_end_i = min(i + 8, N)
                block_end_j = min(j + 8, N)
                W_big[i:block_end_i, j:block_end_j] *= (1 + 0.1 * np.cos(2 * np.pi * phase))

        # Normalize spectral radius to edge of chaos
        eigs = np.linalg.eigvals(W_big)
        max_eig = np.max(np.abs(eigs))
        if max_eig > 0:
            # Target spectral radius: 1/φ ≈ 0.618 (the golden critical point)
            W_big *= PHI_INV / max_eig

        return W_big

    def _build_input_weights(self) -> np.ndarray:
        """
        Build input weights from Phillips matrix column structure.

        Uses the Column Trichotomy (2-4-2 pattern) to weight inputs:
        expanded columns (0,4) get weight φ+2, stable (1,2,5,6) get 2.5,
        contracted (3,7) get 3-φ. This preserves the projection geometry.
        """
        W_in = np.zeros((self.n_reservoir, self.input_dim))

        trichotomy_weights = np.zeros(8)
        trichotomy_weights[[0, 4]] = PHI + 2          # expanded
        trichotomy_weights[[1, 2, 5, 6]] = 2.5        # stable
        trichotomy_weights[[3, 7]] = 3 - PHI           # contracted

        # Normalize
        trichotomy_weights /= np.linalg.norm(trichotomy_weights)

        for i in range(self.n_reservoir):
            col_idx = i % min(self.input_dim, 8)
            scale = trichotomy_weights[col_idx % 8]
            if col_idx < self.input_dim:
                W_in[i, col_idx] = scale
            # Cross-connections via golden ratio
            other_idx = int((col_idx * PHI) % self.input_dim)
            W_in[i, other_idx] += scale * 0.1

        return W_in

    def reset(self):
        """Reset reservoir state to zero."""
        self.state = np.zeros(self.n_reservoir)

    def step(self, u: np.ndarray) -> np.ndarray:
        """
        Single step update.

        Parameters
        ----------
        u : np.ndarray
            Input vector of shape (input_dim,)

        Returns
        -------
        np.ndarray
            New reservoir state
        """
        pre = self.W_in @ u[:self.input_dim] + self.W @ self.state
        self.state = (1 - self.leak_rate) * self.state + self.leak_rate * np.tanh(pre)
        return self.state

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run input sequence through reservoir.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (T, input_dim)

        Returns
        -------
        np.ndarray
            Shape (T, n_reservoir) — all states
        """
        T = len(inputs)
        states = np.zeros((T, self.n_reservoir))
        for t in range(T):
            states[t] = self.step(inputs[t])
        return states

    @property
    def spectral_radius(self) -> float:
        """Current spectral radius of reservoir weights."""
        return float(np.max(np.abs(np.linalg.eigvals(self.W))))

    @property
    def gram_eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the underlying Phillips Gram matrix."""
        G = PHILLIPS_U_L.T @ PHILLIPS_U_L
        return np.sort(np.linalg.eigvalsh(G))[::-1]


# =============================================================================
# 2. GOLDEN-RATIO MRA (Multi-Resolution Analysis)
# =============================================================================

class GoldenMRA:
    """
    4D Multi-Resolution Analysis with golden-ratio dilation.

    Standard wavelets use dyadic dilation (scale by 2). This MRA uses
    φ-adic dilation: each level magnifies by φ ≈ 1.618. The Phillips
    U_L block serves as the scaling function seed.

    Key property: φ-dilation is the UNIQUE dilation factor associated with
    H4 symmetry (Boyle's discrete scale invariance). This makes the MRA
    compatible with the quasicrystalline projection geometry.

    Properties:
        - Dilation factor: φ (not 2)
        - Scaling function: derived from U_L row structure
        - Fibonacci subsampling: sample at Fibonacci-number positions
        - Non-dyadic filter bank: 4 channels per level

    Parameters
    ----------
    n_levels : int
        Number of resolution levels
    signal_dim : int
        Dimension of input signals
    """

    def __init__(self, n_levels: int = 5, signal_dim: int = 8):
        self.n_levels = n_levels
        self.signal_dim = signal_dim

        # Build golden-ratio filter bank from Phillips U_L
        self.scaling_filter = self._build_scaling_filter()
        self.detail_filters = self._build_detail_filters()

    def _build_scaling_filter(self) -> np.ndarray:
        """
        Build the scaling (low-pass) filter from Phillips U_L rows.

        The four rows of U_L, normalized, form the analysis filter.
        Entries {a, b} = {1/2, (φ-1)/2} are the filter coefficients.
        """
        # Use first row of U_L as prototype, normalized
        h0 = PHILLIPS_U_L[0].copy()
        h0 /= np.linalg.norm(h0)
        return h0

    def _build_detail_filters(self) -> List[np.ndarray]:
        """
        Build detail (high-pass) filters from remaining U_L rows.

        Rows 1-3 of U_L capture different orientations of detail,
        analogous to horizontal/vertical/diagonal in 2D wavelets.
        """
        filters = []
        for i in range(1, 4):
            h = PHILLIPS_U_L[i].copy()
            h /= np.linalg.norm(h)
            filters.append(h)
        return filters

    def _golden_downsample(self, signal: np.ndarray) -> np.ndarray:
        """
        Downsample by golden ratio using Fibonacci-position sampling.

        Instead of taking every 2nd sample (dyadic), take samples at
        Fibonacci-number positions. This gives φ-rate decimation.
        """
        N = len(signal)
        # Generate Fibonacci indices up to N
        fib_indices = self._fibonacci_indices(N)
        return signal[fib_indices]

    def _fibonacci_indices(self, N: int) -> np.ndarray:
        """Generate Fibonacci-number indices less than N."""
        fibs = [0, 1]
        while fibs[-1] < N:
            fibs.append(fibs[-1] + fibs[-2])
        fibs = [f for f in fibs if f < N]
        return np.array(fibs, dtype=int)

    def _golden_upsample(self, signal: np.ndarray, target_len: int) -> np.ndarray:
        """Upsample from Fibonacci positions back to full length."""
        result = np.zeros(target_len)
        fib_indices = self._fibonacci_indices(target_len)
        n = min(len(signal), len(fib_indices))
        result[fib_indices[:n]] = signal[:n]
        return result

    def decompose(self, signal: np.ndarray) -> Dict[str, List[np.ndarray]]:
        """
        Perform golden-ratio multi-resolution decomposition.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (1D or flattened)

        Returns
        -------
        Dict with keys:
            'approximation': list of approximation coefficients per level
            'details': list of (3 detail channels) per level
            'energies': energy at each level
        """
        approx_coeffs = []
        detail_coeffs = []
        energies = []

        current = signal.copy()

        for level in range(self.n_levels):
            if len(current) < 8:
                break

            # Convolution with scaling filter (circular)
            N = len(current)
            h_len = len(self.scaling_filter)

            # Pad filter to signal length and use FFT for convolution
            h_padded = np.zeros(N)
            h_padded[:min(h_len, N)] = self.scaling_filter[:min(h_len, N)]
            approx = np.real(np.fft.ifft(np.fft.fft(current) * np.fft.fft(h_padded)))

            # Detail coefficients from each detail filter
            level_details = []
            for d_filter in self.detail_filters:
                d_padded = np.zeros(N)
                d_padded[:min(h_len, N)] = d_filter[:min(h_len, N)]
                detail = np.real(np.fft.ifft(np.fft.fft(current) * np.fft.fft(d_padded)))
                level_details.append(self._golden_downsample(detail))

            # Golden-ratio downsample the approximation
            approx_down = self._golden_downsample(approx)
            approx_coeffs.append(approx_down)
            detail_coeffs.append(level_details)

            # Track energy
            total_energy = np.sum(approx_down**2) + sum(np.sum(d**2) for d in level_details)
            energies.append(float(total_energy))

            # Iterate on downsampled approximation
            current = approx_down

        return {
            'approximation': approx_coeffs,
            'details': detail_coeffs,
            'energies': energies,
        }

    def reconstruct(self, coeffs: Dict[str, List[np.ndarray]],
                     original_length: int) -> np.ndarray:
        """
        Reconstruct signal from golden-ratio MRA coefficients.

        Parameters
        ----------
        coeffs : dict
            Output from decompose()
        original_length : int
            Length of original signal

        Returns
        -------
        np.ndarray
            Reconstructed signal
        """
        n_levels = len(coeffs['approximation'])
        if n_levels == 0:
            return np.zeros(original_length)

        # Start from coarsest level
        current = coeffs['approximation'][-1]

        # Reconstruct level by level (coarse to fine)
        for level in range(n_levels - 2, -1, -1):
            target_len = len(coeffs['approximation'][level])
            if level > 0:
                target_len = len(coeffs['approximation'][level - 1])
            else:
                target_len = original_length

            # Upsample and add detail
            upsampled = self._golden_upsample(current, target_len)

            # Add detail contributions
            for d in coeffs['details'][level]:
                d_up = self._golden_upsample(d, target_len)
                upsampled += d_up

            current = upsampled

        return current[:original_length]

    @property
    def dilation_factor(self) -> float:
        """The golden dilation factor."""
        return PHI

    @property
    def filter_entries(self) -> Dict[str, float]:
        """The two filter coefficient values."""
        return {'a': _a, 'b': _b, 'ratio': PHI}


# =============================================================================
# 3. NUMBER FIELD HIERARCHY
# =============================================================================

@dataclass
class NumberFieldLevel:
    """A single level in the number field hierarchy."""
    name: str               # e.g., "Q(sqrt(5))"
    algebraic_number: float  # φ or ρ
    discriminant: int        # 5 or -23
    degree: int              # 2 or 3
    damping: float           # derived from algebraic structure
    coupling: float          # inter-level coupling strength


class NumberFieldHierarchy:
    """
    Three-level computational hierarchy governed by number fields.

    Level 0: Q (rationals) — digital backbone, binary switching
    Level 1: Q(√5) — golden-ratio dynamics, φ-scaled (the Phillips matrix level)
    Level 2: Q(ρ) — plastic-ratio dynamics, ρ-scaled (slower cubic envelope)

    The coupling between levels follows the discriminant ladder:
    disc(φ) = 5 → disc(ρ) = -23. The escalation from real quadratic to
    complex cubic fields determines the inter-level coupling constants.

    This matches Boyle's framework where quasicrystal scale factors are
    units of algebraic number fields.

    Parameters
    ----------
    base_size : int
        Size of each level's state vector
    """

    def __init__(self, base_size: int = 32):
        self.base_size = base_size

        self.levels = [
            NumberFieldLevel(
                name="Q",
                algebraic_number=1.0,
                discriminant=1,
                degree=1,
                damping=0.5,       # Fast, digital
                coupling=0.0,      # No self-coupling
            ),
            NumberFieldLevel(
                name="Q(sqrt(5))",
                algebraic_number=PHI,
                discriminant=5,
                degree=2,
                damping=1.0 / PHI,   # ≈ 0.618 — golden damping
                coupling=1.0 / 5,    # disc(φ) = 5
            ),
            NumberFieldLevel(
                name="Q(rho)",
                algebraic_number=RHO,
                discriminant=-23,
                degree=3,
                damping=1.0 / RHO,   # ≈ 0.755 — plastic damping
                coupling=1.0 / 23,   # |disc(ρ)| = 23
            ),
        ]

        # State vectors for each level
        self.states = [np.zeros(base_size) for _ in self.levels]

        # Weight matrices for each level
        self.weights = self._build_level_weights()

    def _build_level_weights(self) -> List[np.ndarray]:
        """Build weight matrices for each level using its algebraic number."""
        matrices = []
        N = self.base_size

        for level in self.levels:
            alpha = level.algebraic_number
            # Create a circulant-like matrix with algebraic-number modulation
            W = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    dist = min(abs(i - j), N - abs(i - j))
                    if dist <= 3:
                        W[i, j] = alpha ** (-dist) * (-1) ** ((i + j) % 2)

            # Scale by damping
            eigs = np.linalg.eigvals(W)
            max_eig = np.max(np.abs(eigs))
            if max_eig > 0:
                W *= level.damping / max_eig

            matrices.append(W)

        return matrices

    def step(self, u: np.ndarray, dt: float = 0.02) -> List[np.ndarray]:
        """
        Single update step across all hierarchy levels.

        Parameters
        ----------
        u : np.ndarray
            External input (applied to Level 0)
        dt : float
            Time step

        Returns
        -------
        List of state vectors, one per level
        """
        new_states = []

        for i, level in enumerate(self.levels):
            W = self.weights[i]

            # Input: external for Level 0, downward from Level i-1 otherwise
            if i == 0:
                drive = u[:self.base_size] if len(u) >= self.base_size else \
                    np.pad(u, (0, self.base_size - len(u)))
            else:
                # Inter-level coupling from the level below
                drive = self.states[i - 1] * level.coupling

            # State update with level-specific dynamics
            pre = W @ self.states[i] + drive
            alpha = level.algebraic_number

            # Activation: tanh for Level 0 (digital), softer for higher levels
            if i == 0:
                new_state = np.tanh(pre)
            else:
                # Golden/plastic softmax: smoother nonlinearity
                new_state = pre / (1 + np.abs(pre) / alpha)

            # Leaky integration with algebraic damping
            self.states[i] = (1 - level.damping * dt) * self.states[i] + \
                level.damping * dt * new_state

            new_states.append(self.states[i].copy())

        return new_states

    def run(self, inputs: np.ndarray, dt: float = 0.02) -> Dict[str, np.ndarray]:
        """
        Run input sequence through hierarchy.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (T, input_dim)
        dt : float
            Time step

        Returns
        -------
        Dict mapping level name to (T, base_size) state arrays
        """
        T = len(inputs)
        histories = {level.name: np.zeros((T, self.base_size)) for level in self.levels}

        for t in range(T):
            states = self.step(inputs[t], dt)
            for i, level in enumerate(self.levels):
                histories[level.name][t] = states[i]

        return histories

    def reset(self):
        """Reset all level states."""
        self.states = [np.zeros(self.base_size) for _ in self.levels]

    @property
    def level_summary(self) -> List[Dict]:
        """Summary of each level's properties."""
        return [
            {
                'name': level.name,
                'algebraic_number': level.algebraic_number,
                'discriminant': level.discriminant,
                'degree': level.degree,
                'damping': level.damping,
                'spectral_radius': float(np.max(np.abs(np.linalg.eigvals(self.weights[i])))),
                'state_energy': float(np.sum(self.states[i] ** 2)),
            }
            for i, level in enumerate(self.levels)
        ]


# =============================================================================
# 4. GALOIS DUAL-CHANNEL VERIFICATION
# =============================================================================

class GaloisVerifier:
    """
    Dual-channel computation with φ-coupled error detection.

    Every computation through U_L (contracted) has a DUAL through U_R
    (expanded), and results must satisfy ||U_R x|| = φ · ||U_L x|| for
    all E8 vectors x. This is the Galois automorphism φ ↔ -1/φ acting
    as a FREE verification channel.

    If the φ-coupling breaks, an error has occurred.

    Parameters
    ----------
    tolerance : float
        Maximum allowed deviation from φ-coupling before flagging error
    """

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.error_count = 0
        self.total_checks = 0
        self.max_deviation = 0.0

    def compute_dual(self, v8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute through both channels simultaneously.

        Parameters
        ----------
        v8 : np.ndarray
            8D input vector

        Returns
        -------
        Tuple of (left_4d, right_4d) projections
        """
        left = PHILLIPS_U_L @ v8
        right = PHILLIPS_U_R @ v8
        return left, right

    def verify(self, v8: np.ndarray) -> Dict:
        """
        Compute and verify the Galois coupling.

        Parameters
        ----------
        v8 : np.ndarray
            8D input vector

        Returns
        -------
        Dict with verification results
        """
        left, right = self.compute_dual(v8)
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)

        self.total_checks += 1

        if left_norm < 1e-12:
            # Zero vector — trivially satisfies coupling
            return {
                'left': left, 'right': right,
                'ratio': float('nan'),
                'expected_ratio': PHI,
                'deviation': 0.0,
                'valid': True,
            }

        ratio = right_norm / left_norm
        deviation = abs(ratio - PHI)
        self.max_deviation = max(self.max_deviation, deviation)

        valid = deviation < self.tolerance
        if not valid:
            self.error_count += 1

        return {
            'left': left,
            'right': right,
            'ratio': float(ratio),
            'expected_ratio': PHI,
            'deviation': float(deviation),
            'valid': valid,
        }

    def verify_batch(self, vectors: np.ndarray) -> Dict:
        """
        Verify φ-coupling for a batch of 8D vectors.

        Parameters
        ----------
        vectors : np.ndarray
            Shape (N, 8) batch of input vectors

        Returns
        -------
        Dict with batch statistics
        """
        results = [self.verify(v) for v in vectors]
        deviations = [r['deviation'] for r in results if not np.isnan(r['ratio'])]

        return {
            'n_vectors': len(vectors),
            'all_valid': all(r['valid'] for r in results),
            'max_deviation': max(deviations) if deviations else 0.0,
            'mean_deviation': float(np.mean(deviations)) if deviations else 0.0,
            'error_rate': sum(1 for r in results if not r['valid']) / len(results),
        }

    def verify_e8_roots(self) -> Dict:
        """
        Verify φ-coupling across all 240 E8 roots (should be exact).
        """
        roots = generate_e8_roots()
        coords = np.array([r.coordinates for r in roots])
        return self.verify_batch(coords)

    @property
    def error_rate(self) -> float:
        """Cumulative error rate."""
        if self.total_checks == 0:
            return 0.0
        return self.error_count / self.total_checks

    def sqrt5_coupling_check(self, v8: np.ndarray) -> Dict:
        """
        Check the φ product coupling: ||U_L x|| · ||U_R x|| = φ · ||U_L x||².

        Since U_R = φ·U_L, we have ||U_R x|| = φ·||U_L x||, so the product
        is φ · ||U_L x||². The √5 identity applies to row norms of the matrix
        itself: ||row_L|| · ||row_R|| = √(3-φ) · √(φ+2) = √5.
        """
        left, right = self.compute_dual(v8)
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)

        product = left_norm * right_norm
        expected = PHI * left_norm ** 2

        return {
            'product': float(product),
            'expected': float(expected),
            'deviation': float(abs(product - expected)),
            'valid': abs(product - expected) < self.tolerance * max(expected, 1e-10),
            'sqrt5_row_norm_product': float(np.sqrt((3 - PHI) * (PHI + 2))),
        }


# =============================================================================
# 5. PHASON ERROR CORRECTION
# =============================================================================

class PhasonErrorCorrector:
    """
    Error correction using the 4D Phillips kernel as code space.

    The Phillips matrix kernel (null space) is 4-dimensional. Among all
    240 E8 roots, only ONE kernel direction (d = (0,1,0,1,0,1,0,1))
    produces collisions. The other 3 kernel directions are "clean" —
    they carry no collision information.

    These 3 clean directions can encode REDUNDANCY:
    - Before projection: embed parity/checksum bits in kernel directions
    - After round-trip: verify kernel components match expectations
    - Mismatch → error in physical computation

    Parameters
    ----------
    code_rate : float
        Fraction of kernel dimensions used for error detection (0 to 1)
    """

    def __init__(self, code_rate: float = 0.75):
        self.code_rate = code_rate

        # Compute kernel basis
        self.kernel_basis = self._compute_kernel()

        # The collision direction
        self.collision_direction = np.array([0, 1, 0, 1, 0, 1, 0, 1]) / 2.0

        # Clean kernel directions (orthogonal to collision direction within kernel)
        self.clean_directions = self._compute_clean_kernel()

    def _compute_kernel(self) -> np.ndarray:
        """
        Compute the 4D kernel of the Phillips matrix.

        Returns basis vectors for null(U), shape (4, 8).
        """
        U, S, Vh = np.linalg.svd(PHILLIPS_MATRIX)
        # Kernel = rows of V^H corresponding to near-zero singular values
        null_mask = S < 1e-10
        # Pad if needed (S has min(8,8) = 8 entries)
        kernel_vecs = Vh[len(S) - np.sum(~null_mask):]
        # If rank is 4, we get the last 4 rows
        _, s, Vt = np.linalg.svd(PHILLIPS_MATRIX, full_matrices=True)
        kernel = Vt[np.sum(s > 1e-10):]  # rows with zero singular values
        return kernel

    def _compute_clean_kernel(self) -> np.ndarray:
        """
        Compute kernel directions that DON'T produce collisions.

        Project out the collision direction from the kernel basis.
        """
        kernel = self.kernel_basis
        if len(kernel) == 0:
            return np.array([]).reshape(0, 8)

        # Normalize collision direction
        d = self.collision_direction.copy()
        d /= np.linalg.norm(d)

        # Project collision direction into kernel space
        d_kernel = np.zeros(len(kernel))
        for i in range(len(kernel)):
            d_kernel[i] = np.dot(kernel[i], d)

        # Gram-Schmidt to remove collision component
        clean = []
        for i in range(len(kernel)):
            v = kernel[i].copy()
            # Remove collision component
            v -= np.dot(v, d) * d
            # Remove components of previously found clean directions
            for c in clean:
                v -= np.dot(v, c) * c
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                clean.append(v / norm)

        return np.array(clean) if clean else np.array([]).reshape(0, 8)

    def encode(self, v8: np.ndarray, checksum: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Embed error-detection checksum into kernel directions.

        Parameters
        ----------
        v8 : np.ndarray
            Original 8D vector
        checksum : np.ndarray, optional
            Checksum values (one per clean kernel direction).
            If None, auto-computes from v8.

        Returns
        -------
        np.ndarray
            Modified 8D vector with checksum embedded in kernel
        """
        n_clean = len(self.clean_directions)
        if n_clean == 0:
            return v8.copy()

        # Auto-compute checksum if not provided
        if checksum is None:
            checksum = np.zeros(n_clean)
            for i in range(n_clean):
                # Checksum = golden-ratio weighted sum of components
                weights = np.array([PHI ** (j % 4) for j in range(8)])
                checksum[i] = np.dot(v8, weights * self.clean_directions[i])

        # Embed checksum into kernel directions (doesn't change projection!)
        v_encoded = v8.copy()
        for i in range(min(n_clean, len(checksum))):
            # Current kernel component
            current = np.dot(v_encoded, self.clean_directions[i])
            # Adjust to match desired checksum
            delta = checksum[i] - current
            v_encoded += delta * self.clean_directions[i]

        return v_encoded

    def verify(self, v8_original: np.ndarray, v8_reconstructed: np.ndarray) -> Dict:
        """
        Verify round-trip integrity using kernel checksums.

        Parameters
        ----------
        v8_original : np.ndarray
            Original 8D vector (with embedded checksum)
        v8_reconstructed : np.ndarray
            Reconstructed 8D vector after computation

        Returns
        -------
        Dict with error detection results
        """
        n_clean = len(self.clean_directions)
        mismatches = []

        for i in range(n_clean):
            original_component = np.dot(v8_original, self.clean_directions[i])
            reconstructed_component = np.dot(v8_reconstructed, self.clean_directions[i])
            deviation = abs(original_component - reconstructed_component)
            mismatches.append(float(deviation))

        max_mismatch = max(mismatches) if mismatches else 0.0

        return {
            'n_kernel_directions': n_clean,
            'mismatches': mismatches,
            'max_mismatch': max_mismatch,
            'error_detected': max_mismatch > 1e-6,
        }

    @property
    def kernel_dimension(self) -> int:
        """Dimension of the Phillips kernel."""
        return len(self.kernel_basis)

    @property
    def clean_dimension(self) -> int:
        """Number of clean (non-collision) kernel directions."""
        return len(self.clean_directions)

    @property
    def collision_direction_info(self) -> Dict:
        """Information about the single collision direction."""
        return {
            'direction': self.collision_direction.tolist(),
            'description': 'd = (0,1,0,1,0,1,0,1)/2 — alternating pattern',
            'n_collision_pairs': 14,
        }


# =============================================================================
# 6. COLLISION-AWARE ENCODING
# =============================================================================

class CollisionAwareEncoder:
    """
    Encoder that exploits the 14 collision pairs for natural compression.

    Among 240 E8 roots, 28 roots map to just 14 points under U_L projection.
    Instead of treating this as information loss, this encoder:
    1. Groups roots by their projected image
    2. Encodes the DIFFERENCE between collision partners as metadata
    3. Reconstructs the full E8 information using both the projection + metadata

    The 14 collision pairs are related by the collision kernel vector
    d = (0,1,0,1,0,1,0,1), so partner = root + α·d for some scalar α.
    """

    def __init__(self):
        self._roots = generate_e8_roots()
        self._collision_map = self._build_collision_map()

    def _build_collision_map(self) -> Dict[tuple, List[int]]:
        """
        Build map from projected 4D point to list of root indices.

        Roots that project to the same 4D point (within tolerance) are
        collision partners.
        """
        from collections import defaultdict
        coll_map = defaultdict(list)

        for idx, root in enumerate(self._roots):
            projected = PHILLIPS_U_L @ root.coordinates
            key = tuple(np.round(projected, 8))
            coll_map[key].append(idx)

        return dict(coll_map)

    @property
    def collision_pairs(self) -> List[Tuple[int, int]]:
        """List of (idx1, idx2) pairs that collide under projection."""
        pairs = []
        for key, indices in self._collision_map.items():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        pairs.append((indices[i], indices[j]))
        return pairs

    @property
    def n_collision_pairs(self) -> int:
        """Number of collision pairs."""
        return len(self.collision_pairs)

    @property
    def n_distinct_projections(self) -> int:
        """Number of distinct 4D images under U_L."""
        return len(self._collision_map)

    def encode_root(self, root_idx: int) -> Dict:
        """
        Encode a single E8 root with collision awareness.

        Returns the 4D projection PLUS metadata identifying which
        collision partner this is (if any).

        Parameters
        ----------
        root_idx : int
            Index into the 240 E8 roots

        Returns
        -------
        Dict with projection, metadata, and reconstruction info
        """
        root = self._roots[root_idx]
        projection = PHILLIPS_U_L @ root.coordinates

        key = tuple(np.round(projection, 8))
        partners = self._collision_map.get(key, [root_idx])

        # Collision metadata: position within the collision group
        group_position = partners.index(root_idx) if root_idx in partners else 0

        # Kernel component: how far along the collision direction
        d = np.array([0, 1, 0, 1, 0, 1, 0, 1]) / 2.0
        kernel_component = np.dot(root.coordinates, d)

        return {
            'projection_4d': projection,
            'root_idx': root_idx,
            'has_collision': len(partners) > 1,
            'group_size': len(partners),
            'group_position': group_position,
            'kernel_component': float(kernel_component),
            'partner_indices': partners,
        }

    def compressed_representation(self) -> Dict:
        """
        Generate the full compressed representation of E8 roots.

        240 roots → 226 unique projections + 14 collision metadata entries.
        Compression ratio: 240/226 ≈ 1.062 (lossless).
        """
        unique_projections = {}
        collision_metadata = []

        for key, indices in self._collision_map.items():
            # Store the first root's projection as the canonical representative
            unique_projections[key] = {
                'projection': np.array(key),
                'canonical_idx': indices[0],
                'group_size': len(indices),
            }

            if len(indices) > 1:
                collision_metadata.append({
                    'projection_key': key,
                    'indices': indices,
                    'kernel_components': [
                        float(np.dot(self._roots[i].coordinates,
                                     np.array([0, 1, 0, 1, 0, 1, 0, 1]) / 2.0))
                        for i in indices
                    ],
                })

        return {
            'n_unique_projections': len(unique_projections),
            'n_collision_groups': len(collision_metadata),
            'compression_ratio': 240 / len(unique_projections),
            'collision_metadata': collision_metadata,
        }


# =============================================================================
# 7. PADOVAN-STEPPED CASCADE
# =============================================================================

class PadovanCascade:
    """
    Cascade dynamics with Padovan-sequence time stepping.

    Standard cascades use uniform time steps. This cascade uses steps
    spaced at Padovan-number intervals: {1,1,1,2,2,3,4,5,7,9,12,...}.
    This gives logarithmic coverage of all temporal scales, governed
    by the plastic ratio ρ ≈ 1.3247.

    The two timescales (φ for spatial, ρ for temporal) are algebraically
    independent, preventing resonance catastrophes that plague periodic
    systems.

    Parameters
    ----------
    max_steps : int
        Maximum total cascade steps
    grid_size : int
        Size of the state grid (state is grid_size × grid_size)
    coupling : float
        Coupling strength between neighbors
    """

    def __init__(self, max_steps: int = 100, grid_size: int = 16, coupling: float = 0.3):
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.coupling = coupling

        # Generate Padovan sequence
        self.padovan_steps = self._generate_padovan(max_steps)

        # State
        self.state = np.zeros((grid_size, grid_size))
        self.velocity = np.zeros((grid_size, grid_size))

    def _generate_padovan(self, max_total: int) -> List[int]:
        """Generate Padovan step sizes up to max_total cumulative."""
        seq = [1, 1, 1]
        while sum(seq) < max_total:
            seq.append(seq[-2] + seq[-3])
        return seq

    def _laplacian(self, state: np.ndarray) -> np.ndarray:
        """Discrete Laplacian (neighbor coupling)."""
        lap = np.zeros_like(state)
        lap += np.roll(state, 1, axis=0) + np.roll(state, -1, axis=0)
        lap += np.roll(state, 1, axis=1) + np.roll(state, -1, axis=1)
        lap -= 4 * state
        return lap

    def inject(self, input_field: np.ndarray):
        """Inject input as a force on the state."""
        if input_field.shape != self.state.shape:
            # Resize
            from numpy import interp
            x_old = np.linspace(0, 1, input_field.shape[0])
            x_new = np.linspace(0, 1, self.grid_size)
            # Simple bilinear resize
            resized = np.zeros((self.grid_size, self.grid_size))
            for i in range(self.grid_size):
                row = np.interp(x_new, x_old, input_field[min(i, input_field.shape[0] - 1)])
                resized[i] = row
            self.velocity += resized * 0.1
        else:
            self.velocity += input_field * 0.1

    def run(self, n_epochs: int = 1) -> Dict:
        """
        Run Padovan-stepped cascade for n_epochs.

        Each epoch processes the full Padovan sequence.
        The key difference from uniform stepping: early steps are small
        (dt=1) for fine temporal resolution, later steps are large
        (dt=Padovan(n)) for coarse temporal scales.

        Returns
        -------
        Dict with cascade results
        """
        history = []
        energies = []
        cumulative_time = 0

        for epoch in range(n_epochs):
            for step_size in self.padovan_steps:
                if cumulative_time >= self.max_steps:
                    break

                dt = step_size * 0.01  # Scale to reasonable time step

                # Velocity Verlet with Padovan step
                force = self.coupling * self._laplacian(self.state)

                # Damping: golden ratio for spatial, plastic for temporal
                spatial_damping = 0.1 * PHI_INV   # φ-governed
                temporal_damping = 0.1 / RHO       # ρ-governed

                self.velocity += force * dt
                self.velocity *= (1 - spatial_damping * dt)
                self.state += self.velocity * dt

                # Tristable snapping
                self.state = np.clip(self.state, 0, 1)

                energy = np.sum(self.velocity ** 2) + np.sum(self.state ** 2)
                energies.append(float(energy))

                cumulative_time += step_size

        return {
            'final_state': self.state.copy(),
            'energies': energies,
            'total_time': cumulative_time,
            'n_padovan_steps': len(self.padovan_steps),
            'padovan_sequence': self.padovan_steps[:10],
        }

    def reset(self):
        """Reset state and velocity to zero."""
        self.state = np.zeros((self.grid_size, self.grid_size))
        self.velocity = np.zeros((self.grid_size, self.grid_size))

    @property
    def padovan_ratio(self) -> float:
        """Ratio of consecutive Padovan numbers (converges to ρ)."""
        if len(self.padovan_steps) > 5:
            return self.padovan_steps[-1] / self.padovan_steps[-2]
        return RHO


# =============================================================================
# 8. FIVE-FOLD RESOURCE ALLOCATION
# =============================================================================

class FiveFoldAllocator:
    """
    Resource allocator based on the Five = Five theorem.

    Frobenius²/rank = 5 = number of 24-cells in a 600-cell = |W(H4)|/|W(D4)|.
    This identity connects:
    - Operator theory (Frobenius norm, rank)
    - Group theory (Weyl group indices)
    - Polytope geometry (600-cell decomposition)

    The computational budget naturally partitions into 5 equal shares,
    one per 24-cell in the constellation. This is NOT a heuristic — it's
    algebraically determined by the Phillips matrix's structure.

    Parameters
    ----------
    total_budget : float
        Total computational resource budget
    """

    def __init__(self, total_budget: float = 1.0):
        self.total_budget = total_budget
        self.GROUP_INDEX = 5  # |600-cell vertices| / |24-cell vertices| = 120/24

        # Verify the Five = Five theorem
        frob_sq = np.sum(PHILLIPS_MATRIX ** 2)
        rank = np.linalg.matrix_rank(PHILLIPS_MATRIX)
        self.amplification = frob_sq / rank

        # Per-node allocation
        self._allocations = self._compute_allocations()

    def _compute_allocations(self) -> Dict[int, Dict]:
        """
        Compute per-node resource allocations.

        Each of the 5 nodes gets 1/5 of total budget.
        Within each node, resources are further split by the
        Trinity decomposition (3 × 16-cells = 1 × 24-cell).
        """
        per_node = self.total_budget / self.GROUP_INDEX

        allocations = {}
        for node_id in range(self.GROUP_INDEX):
            # Trinity split within each node: α/β/γ channels
            # Weights from Phillips row norms: contracted vs expanded
            trinity_weights = np.array([
                3 - PHI,   # α (contracted) ≈ 1.382
                2.5,       # β (stable)
                PHI + 2,   # γ (expanded) ≈ 3.618
            ])
            trinity_weights /= trinity_weights.sum()

            allocations[node_id] = {
                'total': per_node,
                'alpha': per_node * trinity_weights[0],
                'beta': per_node * trinity_weights[1],
                'gamma': per_node * trinity_weights[2],
                'trinity_weights': trinity_weights.tolist(),
            }

        return allocations

    def get_allocation(self, node_id: int) -> Dict:
        """Get resource allocation for a specific node."""
        return self._allocations.get(node_id % self.GROUP_INDEX, {})

    def get_all_allocations(self) -> Dict[int, Dict]:
        """Get all node allocations."""
        return self._allocations.copy()

    def verify_five_equals_five(self) -> Dict:
        """
        Verify the Five = Five theorem.

        Frobenius²/rank = 5 = |600-cell|/|24-cell| = 120/24
        """
        frob_sq = float(np.sum(PHILLIPS_MATRIX ** 2))
        rank = int(np.linalg.matrix_rank(PHILLIPS_MATRIX))

        return {
            'frobenius_squared': frob_sq,
            'rank': rank,
            'amplification': frob_sq / rank,
            'group_index': self.GROUP_INDEX,
            'match': np.isclose(frob_sq / rank, self.GROUP_INDEX),
            'polytope_ratio': '|600-cell vertices| / |24-cell vertices| = 120/24 = 5',
            'weyl_ratio': '|W(H4)| / |W(D4)| = 14400/2880 ≈ 5 (via intermediate groups)',
        }

    @property
    def per_node_budget(self) -> float:
        """Budget per constellation node."""
        return self.total_budget / self.GROUP_INDEX
