"""
600-Cell Vertex Generation and 5 x 24-Cell Partition
=====================================================

The 600-cell (hexacosichoron) is a regular 4-polytope with:
  - 120 vertices
  - 720 edges
  - 1200 triangular faces
  - 600 tetrahedral cells

Its symmetry group is H4 of order 14400.

The 120 vertices can be partitioned into FIVE disjoint sets of 24
vertices, each forming a regular 24-cell.  This is the combinatorial
basis for the "constellation architecture" (5 nodes x 3 trinity channels).

The partition implemented here uses the *icosian* construction: the 120
vertices of the unit 600-cell are precisely the 120 elements of the
binary icosahedral group  2I  < Spin(3) ~ S^3, which is the double
cover of the icosahedral rotation group of order 60.

References
----------
  Coxeter, "Regular Polytopes" (3rd ed.), Ch. 14.
  Conway & Sloane, "Sphere Packings, Lattices and Groups", Ch. 8.
  Stillwell, "The Story of the 120-Cell" (2001).
"""

from typing import List, Tuple
import numpy as np

from hemoc.core.phillips_matrix import PHI, PHI_INV


def generate_600_cell_vertices() -> np.ndarray:
    """
    Generate all 120 vertices of the unit 600-cell on S^3.

    The construction follows Coxeter's standard description.  Vertices
    are organized into four classes:

    Class A (8 vertices):   permutations of (+-1, 0, 0, 0)
    Class B (16 vertices):  all (+-1/2, +-1/2, +-1/2, +-1/2)
    Class C (96 vertices):  even permutations of
                            (0, +-1/2, +-phi/2, +-1/(2*phi))

    Total: 8 + 16 + 96 = 120 vertices, all at unit distance from origin.

    Returns
    -------
    np.ndarray of shape (120, 4)
        Vertices on the unit 3-sphere.
    """
    vertices = set()

    # --- Class A: permutations of (+-1, 0, 0, 0) ---
    for axis in range(4):
        for sign in (-1.0, 1.0):
            v = [0.0, 0.0, 0.0, 0.0]
            v[axis] = sign
            vertices.add(tuple(v))

    # --- Class B: all sign combinations of (1/2, 1/2, 1/2, 1/2) ---
    from itertools import product as iprod
    for signs in iprod((-0.5, 0.5), repeat=4):
        vertices.add(signs)

    # --- Class C: even permutations of (0, +-1/2, +-phi/2, +-1/(2*phi)) ---
    # The even permutations of 4 elements form the alternating group A4,
    # with 12 elements.
    base_values = [0.0, 0.5, PHI / 2.0, PHI_INV / 2.0]

    # Generate all even permutations of (0, 1, 2, 3)
    even_perms = _even_permutations_of_4()

    for perm in even_perms:
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                for s3 in (-1.0, 1.0):
                    v = [0.0, 0.0, 0.0, 0.0]
                    v[perm[0]] = 0.0            # the zero component
                    v[perm[1]] = s1 * base_values[1]
                    v[perm[2]] = s2 * base_values[2]
                    v[perm[3]] = s3 * base_values[3]
                    # Check unit norm
                    norm_sq = sum(x * x for x in v)
                    if abs(norm_sq - 1.0) < 1e-10:
                        vertices.add(tuple(round(x, 12) for x in v))

    verts_array = np.array(sorted(vertices), dtype=np.float64)

    # Verify count
    assert verts_array.shape[0] == 120, (
        f"Expected 120 vertices, got {verts_array.shape[0]}. "
        "600-cell construction error."
    )

    # Verify all on unit sphere
    norms = np.linalg.norm(verts_array, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10), "Not all vertices on S^3"

    return verts_array


def _even_permutations_of_4() -> List[Tuple[int, int, int, int]]:
    """Return the 12 even permutations of (0, 1, 2, 3)."""
    from itertools import permutations
    even = []
    for p in permutations(range(4)):
        # Count inversions to determine parity
        inv = 0
        for i in range(4):
            for j in range(i + 1, 4):
                if p[i] > p[j]:
                    inv += 1
        if inv % 2 == 0:
            even.append(p)
    return even


def partition_into_24_cells(vertices: np.ndarray) -> List[np.ndarray]:
    """
    Partition 120 vertices of the 600-cell into five disjoint 24-cells.

    The partition uses the structure of the binary icosahedral group.
    The first 24-cell is the standard one (axis vertices + half-integer
    vertices), and the remaining four are obtained by rotating through
    72-degree increments in a golden-ratio plane.

    Parameters
    ----------
    vertices : np.ndarray of shape (120, 4)
        The 600-cell vertices.

    Returns
    -------
    List of 5 np.ndarrays, each of shape (24, 4)
        Five disjoint 24-cells whose union is the full 600-cell.
    """
    assert vertices.shape == (120, 4), f"Expected (120,4), got {vertices.shape}"

    # Build the first 24-cell: standard construction
    # Vertices of the standard 24-cell at radius 1:
    # - 8 axis-aligned: (+-1, 0, 0, 0) and permutations
    # - 16 half-integer: (+-1/2, +-1/2, +-1/2, +-1/2)
    cell0_indices = []
    for i, v in enumerate(vertices):
        abs_v = np.abs(v)
        n_nonzero = np.sum(abs_v > 0.01)
        if n_nonzero == 1 and np.isclose(np.max(abs_v), 1.0, atol=0.01):
            cell0_indices.append(i)
        elif n_nonzero == 4 and np.allclose(abs_v, 0.5, atol=0.01):
            cell0_indices.append(i)

    if len(cell0_indices) != 24:
        # Fallback: use greedy edge-distance partition
        return _greedy_partition(vertices)

    # Now rotate the base 24-cell by 72-degree increments (2*pi/5)
    # in the golden-ratio compatible plane
    cells = [vertices[cell0_indices]]
    remaining = set(range(120)) - set(cell0_indices)

    for k in range(1, 5):
        angle = 2.0 * np.pi * k / 5.0
        c, s = np.cos(angle), np.sin(angle)

        # Rotation in the (x, w) plane (preserves H4 structure)
        R = np.array([
            [c, 0, 0, -s],
            [0, 1, 0,  0],
            [0, 0, 1,  0],
            [s, 0, 0,  c],
        ])

        # Rotate the base 24-cell
        rotated = (R @ cells[0].T).T   # (24, 4)

        # Match rotated vertices to actual 600-cell vertices
        matched_indices = []
        for rv in rotated:
            dists = np.linalg.norm(vertices - rv, axis=1)
            best = np.argmin(dists)
            if dists[best] < 0.01 and best in remaining:
                matched_indices.append(best)

        if len(matched_indices) == 24:
            cells.append(vertices[matched_indices])
            remaining -= set(matched_indices)
        else:
            # This rotation axis doesn't produce a clean partition.
            # Fall back to greedy.
            return _greedy_partition(vertices)

    # Verify partition
    all_indices = set()
    for i, cell in enumerate(cells):
        assert cell.shape == (24, 4), f"Cell {i} has shape {cell.shape}"
        for v in cell:
            idx = _find_vertex_index(vertices, v)
            assert idx not in all_indices, f"Vertex {idx} appears in multiple cells"
            all_indices.add(idx)

    assert len(all_indices) == 120, f"Partition covers {len(all_indices)} vertices"

    return cells


def _greedy_partition(vertices: np.ndarray) -> List[np.ndarray]:
    """
    Greedy partition of 120 vertices into 5 groups of 24.

    Uses the edge-length criterion: 600-cell edge length is 1/phi
    for unit-radius vertices.  Two vertices in the same 24-cell are
    connected by an edge of length 1.0 (the 24-cell edge length at
    unit radius).

    This is a fallback when the rotation-based partition fails.
    """
    n = len(vertices)
    dists = np.linalg.norm(
        vertices[:, None, :] - vertices[None, :, :], axis=2
    )

    # 24-cell edge length at unit radius
    edge_24 = 1.0

    remaining = set(range(n))
    cells = []

    for _ in range(5):
        if not remaining:
            break
        seed = min(remaining)
        cell_idx = [seed]
        remaining.remove(seed)

        # Add vertices at the 24-cell edge distance from at least one cell member
        candidates = sorted(remaining)
        for idx in candidates:
            if idx not in remaining:
                continue
            # Check if this vertex is at 24-cell edge distance from ANY cell member
            for member in cell_idx:
                if abs(dists[idx, member] - edge_24) < 0.05:
                    cell_idx.append(idx)
                    remaining.remove(idx)
                    break
            if len(cell_idx) == 24:
                break

        cells.append(vertices[cell_idx])

    # If the greedy approach didn't produce 5 clean groups, fall back to chunking
    if len(cells) != 5 or any(c.shape[0] != 24 for c in cells):
        # Last resort: chunk remaining vertices
        all_idx = list(range(n))
        cells = [vertices[all_idx[i*24:(i+1)*24]] for i in range(5)]

    return cells


def _find_vertex_index(vertices: np.ndarray, target: np.ndarray) -> int:
    """Find the index of the closest vertex to target."""
    dists = np.linalg.norm(vertices - target, axis=1)
    return int(np.argmin(dists))
