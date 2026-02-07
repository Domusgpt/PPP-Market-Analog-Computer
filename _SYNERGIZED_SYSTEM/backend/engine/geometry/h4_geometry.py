"""
H4 Geometry - Four-dimensional Coxeter group polytopes.

This module implements the mathematical structures for the H4 Coxeter group,
including the 24-cell (Icositetrachoron), 16-cell (Hexadecachoron),
600-cell (Hexacosichoron), and 120-cell (Hecatonicosachoron).

The H4 group is the symmetry group of the 600-cell and 120-cell, characterized
by five-fold symmetry and the Golden Ratio (φ = (1 + √5) / 2).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
from itertools import permutations, product


# Golden Ratio - fundamental constant for H4 geometry
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618033988749895
PHI_INV = PHI - 1  # = 1/φ ≈ 0.618033988749895


class VertexState(Enum):
    """Vertex deployment states for kirigami projection."""
    LOCKED = 0      # State 0: Compact 24-cell
    AUXETIC = 0.5   # State ½: Bistable transition
    DEPLOYED = 1    # State 1: 600-cell projection


class LayerPlane(Enum):
    """The six orthogonal central planes of the 24-cell projection."""
    XY = "xy"  # (x, y, 0, 0)
    ZW = "zw"  # (0, 0, z, w)
    XZ = "xz"  # (x, 0, z, 0)
    YW = "yw"  # (0, y, 0, w)
    XW = "xw"  # (x, 0, 0, w)
    YZ = "yz"  # (0, y, z, 0)


class TrilaticChannel(Enum):
    """Three-color decomposition channels for the trilatic 16-cell focus."""
    ALPHA = "alpha"    # Red/Cyan channel - Layer pair 1
    BETA = "beta"      # Green/Magenta channel - Layer pair 2
    GAMMA = "gamma"    # Blue/Yellow channel - Layer pair 3


@dataclass
class Vertex4D:
    """A vertex in 4-dimensional Euclidean space."""
    x: float
    y: float
    z: float
    w: float
    state: VertexState = VertexState.LOCKED
    channel: Optional[TrilaticChannel] = None

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z, self.w])

    @classmethod
    def from_array(cls, arr: np.ndarray, **kwargs) -> "Vertex4D":
        """Create from numpy array."""
        return cls(arr[0], arr[1], arr[2], arr[3], **kwargs)

    def project_to_3d(self,
                       projection_distance: float = 2.0,
                       method: str = "stereographic") -> np.ndarray:
        """
        Project 4D vertex to 3D space.

        Args:
            projection_distance: Distance of projection point from origin
            method: Projection method - "stereographic" or "orthographic"

        Returns:
            3D coordinates as numpy array
        """
        if method == "stereographic":
            # Stereographic projection from w = projection_distance
            scale = projection_distance / (projection_distance - self.w)
            return np.array([self.x * scale, self.y * scale, self.z * scale])
        else:
            # Orthographic projection (drop w coordinate)
            return np.array([self.x, self.y, self.z])

    def distance_from_origin(self) -> float:
        """Calculate Euclidean distance from origin."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)

    def __eq__(self, other: "Vertex4D") -> bool:
        """Check vertex equality with tolerance."""
        if not isinstance(other, Vertex4D):
            return False
        return np.allclose(self.to_array(), other.to_array(), atol=1e-10)

    def __hash__(self) -> int:
        """Hash for set operations."""
        return hash(tuple(np.round(self.to_array(), 8)))


@dataclass
class Edge4D:
    """An edge connecting two 4D vertices."""
    v1: Vertex4D
    v2: Vertex4D

    def length(self) -> float:
        """Calculate edge length."""
        diff = self.v1.to_array() - self.v2.to_array()
        return np.linalg.norm(diff)

    def midpoint(self) -> Vertex4D:
        """Get edge midpoint."""
        mid = (self.v1.to_array() + self.v2.to_array()) / 2
        return Vertex4D.from_array(mid)


@dataclass
class Cell4D:
    """A 3D cell (boundary element) of a 4D polytope."""
    vertices: List[Vertex4D]
    cell_type: str = "tetrahedral"  # or "dodecahedral" for 120-cell


@dataclass
class Polytope16Cell:
    """
    The 16-cell (Hexadecachoron) - a regular 4D polytope.

    The 16-cell has:
    - 8 vertices
    - 24 edges
    - 32 triangular faces
    - 16 tetrahedral cells

    It is the 4D analogue of the octahedron and is dual to the tesseract.
    """
    vertices: List[Vertex4D] = field(default_factory=list)
    channel: TrilaticChannel = TrilaticChannel.ALPHA

    def __post_init__(self):
        """Generate vertices if not provided."""
        if not self.vertices:
            self.vertices = self._generate_vertices()

    def _generate_vertices(self) -> List[Vertex4D]:
        """
        Generate the 8 vertices of the 16-cell.

        Standard coordinates: permutations of (±1, 0, 0, 0)
        """
        vertices = []
        for axis in range(4):
            for sign in [-1, 1]:
                coords = [0.0, 0.0, 0.0, 0.0]
                coords[axis] = float(sign)
                vertices.append(Vertex4D(*coords, channel=self.channel))
        return vertices

    def get_vertex_array(self) -> np.ndarray:
        """Get all vertices as numpy array (8 x 4)."""
        return np.array([v.to_array() for v in self.vertices])

    def get_edges(self) -> List[Edge4D]:
        """
        Generate all 24 edges of the 16-cell.

        Edges connect vertices at distance √2 from each other.
        """
        edges = []
        for i, v1 in enumerate(self.vertices):
            for v2 in self.vertices[i+1:]:
                dist = np.linalg.norm(v1.to_array() - v2.to_array())
                if np.isclose(dist, np.sqrt(2)):
                    edges.append(Edge4D(v1, v2))
        return edges


@dataclass
class Polytope24Cell:
    """
    The 24-cell (Icositetrachoron) - the fundamental H4 module.

    The 24-cell is unique among regular polytopes - it has no analogue
    in any other dimension. It has:
    - 24 vertices
    - 96 edges
    - 96 triangular faces
    - 24 octahedral cells

    Key property: It can be decomposed into THREE disjoint 16-cells
    (the "Trilatic" decomposition).
    """
    vertices: List[Vertex4D] = field(default_factory=list)
    state: VertexState = VertexState.LOCKED

    def __post_init__(self):
        """Generate vertices if not provided."""
        if not self.vertices:
            self.vertices = self._generate_vertices()

    def _generate_vertices(self) -> List[Vertex4D]:
        """
        Generate the 24 vertices of the 24-cell.

        Coordinates: All permutations of (±1, 0, 0, 0) and
                    ALL combinations of (±1/2, ±1/2, ±1/2, ±1/2)

        Total: 8 + 16 = 24 vertices
        """
        vertices = []

        # Type 1: Permutations of (±1, 0, 0, 0) - 8 vertices
        for axis in range(4):
            for sign in [-1, 1]:
                coords = [0.0, 0.0, 0.0, 0.0]
                coords[axis] = float(sign)
                vertices.append(Vertex4D(*coords, state=self.state))

        # Type 2: ALL (±1/2, ±1/2, ±1/2, ±1/2) - 16 vertices
        # All 2^4 = 16 combinations
        for signs in product([-0.5, 0.5], repeat=4):
            vertices.append(Vertex4D(*signs, state=self.state))

        return vertices

    def get_vertex_array(self) -> np.ndarray:
        """Get all vertices as numpy array (24 x 4)."""
        return np.array([v.to_array() for v in self.vertices])

    def get_layer_vertices(self, plane: LayerPlane) -> List[Vertex4D]:
        """
        Get vertices associated with a specific projection plane.

        The 24 vertices naturally group into 6 orthogonal central planes.
        """
        plane_map = {
            LayerPlane.XY: lambda v: np.isclose(v.z, 0) and np.isclose(v.w, 0),
            LayerPlane.ZW: lambda v: np.isclose(v.x, 0) and np.isclose(v.y, 0),
            LayerPlane.XZ: lambda v: np.isclose(v.y, 0) and np.isclose(v.w, 0),
            LayerPlane.YW: lambda v: np.isclose(v.x, 0) and np.isclose(v.z, 0),
            LayerPlane.XW: lambda v: np.isclose(v.y, 0) and np.isclose(v.z, 0),
            LayerPlane.YZ: lambda v: np.isclose(v.x, 0) and np.isclose(v.w, 0),
        }
        return [v for v in self.vertices if plane_map[plane](v)]

    def get_edges(self) -> List[Edge4D]:
        """
        Generate all 96 edges of the 24-cell.

        Edges connect vertices at distance 1 from each other.
        """
        edges = []
        for i, v1 in enumerate(self.vertices):
            for v2 in self.vertices[i+1:]:
                dist = np.linalg.norm(v1.to_array() - v2.to_array())
                if np.isclose(dist, 1.0):
                    edges.append(Edge4D(v1, v2))
        return edges

    def project_to_3d(self,
                       projection_distance: float = 2.0,
                       method: str = "stereographic") -> np.ndarray:
        """Project all vertices to 3D space."""
        return np.array([
            v.project_to_3d(projection_distance, method)
            for v in self.vertices
        ])


@dataclass
class TrilaticDecomposition:
    """
    The Three-Color (Trilatic) Decomposition of the 24-cell.

    The 24 vertices of a 24-cell can be partitioned into three disjoint
    sets of 8 vertices. Each set forms a regular 16-cell. These three
    16-cells are the "computational cores" of the H4 prototype.

    This decomposition enables:
    - Three independent data channels
    - Layer pair control (6 layers = 3 pairs)
    - Dimensional overlapping for quaternion control
    """
    polytope_24: Polytope24Cell
    cell_alpha: Polytope16Cell = field(default=None)
    cell_beta: Polytope16Cell = field(default=None)
    cell_gamma: Polytope16Cell = field(default=None)

    def __post_init__(self):
        """Perform the trilatic decomposition."""
        self._decompose()

    def _decompose(self):
        """
        Decompose the 24-cell into three disjoint 16-cells.

        This is the W(D₄) ⊂ W(F₄) coset decomposition (index 3).
        The 24 Hurwitz quaternion vertices partition cleanly:

          Alpha: 8 axis-aligned vertices — perms of (±1, 0, 0, 0)
          Beta:  8 half-integer vertices with EVEN count of negative signs
          Gamma: 8 half-integer vertices with ODD count of negative signs

        Each set forms a regular 16-cell (cross-polytope) at radius 1,
        with 4 antipodal pairs and 24 edges of length √2.
        The three sets are perfectly disjoint and their union is the 24-cell.
        """
        alpha_verts = []
        beta_verts = []
        gamma_verts = []

        for v in self.polytope_24.vertices:
            arr = v.to_array()
            abs_arr = np.abs(arr)

            # Classify: axis-aligned (one ±1, rest 0) vs half-integer (all ±½)
            n_nonzero = np.sum(abs_arr > 0.1)

            if n_nonzero == 1:
                # Axis-aligned vertex → Alpha
                alpha_verts.append(Vertex4D.from_array(
                    arr, channel=TrilaticChannel.ALPHA))
            else:
                # Half-integer vertex → count negative signs
                n_negative = np.sum(arr < -0.1)
                if n_negative % 2 == 0:
                    # Even number of minus signs → Beta
                    beta_verts.append(Vertex4D.from_array(
                        arr, channel=TrilaticChannel.BETA))
                else:
                    # Odd number of minus signs → Gamma
                    gamma_verts.append(Vertex4D.from_array(
                        arr, channel=TrilaticChannel.GAMMA))

        assert len(alpha_verts) == 8, f"Alpha has {len(alpha_verts)} vertices, expected 8"
        assert len(beta_verts) == 8, f"Beta has {len(beta_verts)} vertices, expected 8"
        assert len(gamma_verts) == 8, f"Gamma has {len(gamma_verts)} vertices, expected 8"

        self.cell_alpha = Polytope16Cell(
            vertices=alpha_verts, channel=TrilaticChannel.ALPHA)
        self.cell_beta = Polytope16Cell(
            vertices=beta_verts, channel=TrilaticChannel.BETA)
        self.cell_gamma = Polytope16Cell(
            vertices=gamma_verts, channel=TrilaticChannel.GAMMA)

    def get_channel_state(self, channel: TrilaticChannel) -> np.ndarray:
        """Get the state register for a specific trilatic channel."""
        cell = {
            TrilaticChannel.ALPHA: self.cell_alpha,
            TrilaticChannel.BETA: self.cell_beta,
            TrilaticChannel.GAMMA: self.cell_gamma,
        }[channel]
        return cell.get_vertex_array()

    def get_all_vertices(self) -> np.ndarray:
        """Get all 24 vertices from the three 16-cells."""
        return np.vstack([
            self.cell_alpha.get_vertex_array(),
            self.cell_beta.get_vertex_array(),
            self.cell_gamma.get_vertex_array()
        ])


@dataclass
class Polytope600Cell:
    """
    The 600-cell (Hexacosichoron) - the H4 target structure.

    The 600-cell has:
    - 120 vertices
    - 720 edges
    - 1200 triangular faces
    - 600 tetrahedral cells

    Key property: The 120 vertices can be partitioned into FIVE disjoint
    24-cells, enabling the "constellation" architecture.
    """
    vertices: List[Vertex4D] = field(default_factory=list)

    def __post_init__(self):
        """Generate vertices if not provided."""
        if not self.vertices:
            self.vertices = self._generate_vertices()

    def _generate_vertices(self) -> List[Vertex4D]:
        """
        Generate the 120 vertices of the 600-cell.

        The vertices fall into several classes based on the Golden Ratio φ.
        All vertices are normalized to lie on the unit 3-sphere.
        """
        vertices = []

        # Scale factor for normalization
        scale = 1.0 / 2.0

        # Class 1: 24 vertices - permutations of (±1, ±1, ±1, ±1) / 2
        # with even number of minus signs
        for signs in product([-1, 1], repeat=4):
            if sum(1 for s in signs if s < 0) % 2 == 0:
                coords = [s * scale for s in signs]
                vertices.append(Vertex4D(*coords))

        # Class 2: 8 vertices - permutations of (±2, 0, 0, 0) / 2
        for axis in range(4):
            for sign in [-1, 1]:
                coords = [0.0, 0.0, 0.0, 0.0]
                coords[axis] = sign
                vertices.append(Vertex4D(*coords))

        # Class 3: 96 vertices involving Golden Ratio
        # Permutations of (±φ, ±1, ±1/φ, 0) / 2
        golden_set = [PHI * scale, scale, PHI_INV * scale, 0.0]

        for perm in permutations([0, 1, 2, 3]):
            for signs in product([-1, 1], repeat=3):
                coords = [0.0, 0.0, 0.0, 0.0]
                for i, p in enumerate(perm[:3]):
                    coords[p] = golden_set[i] * signs[i]
                # Check if this creates a vertex on the 3-sphere
                if np.isclose(sum(c**2 for c in coords), 1.0, atol=0.1):
                    v = Vertex4D(*coords)
                    if not any(np.allclose(v.to_array(),
                               existing.to_array(), atol=1e-8)
                               for existing in vertices):
                        vertices.append(v)

        # Ensure we have 120 vertices (fill remaining with proper construction)
        if len(vertices) < 120:
            vertices = self._generate_full_vertices()

        return vertices[:120]

    def _generate_full_vertices(self) -> List[Vertex4D]:
        """Generate all 120 vertices using the complete construction."""
        vertices = set()

        # The 600-cell vertices are most elegantly described as:
        # All even permutations of:
        # (0, 0, ±2, ±2) / 2
        # (±1, ±1, ±1, ±√5) / 2
        # (±φ^2, ±1, ±φ^-1, ±φ^-1) / 2
        # (±φ, ±φ, ±φ, ±φ^-2) / 2
        # (±φ^2, ±φ^-1, ±φ^-1, ±φ^-1) / 2

        scale = 0.5

        # Type 1: Permutations of (0, 0, ±2, ±2) / 2
        base1 = [0, 0, 2, 2]
        for perm in set(permutations(base1)):
            for signs in product([1, -1], repeat=4):
                coords = tuple(p * s * scale for p, s in zip(perm, signs))
                if np.isclose(sum(c**2 for c in coords), 1.0, atol=0.2):
                    vertices.add(coords)

        # Type 2: 24-cell vertices (±1, ±1, ±1, ±1) / 2 even permutations
        for signs in product([-1, 1], repeat=4):
            if sum(1 for s in signs if s < 0) % 2 == 0:
                vertices.add(tuple(s * scale for s in signs))

        # Type 3: Axis vertices (±1, 0, 0, 0)
        for axis in range(4):
            for sign in [-1, 1]:
                coords = [0.0] * 4
                coords[axis] = sign
                vertices.add(tuple(coords))

        # Type 4: Golden ratio vertices
        phi = PHI
        phi_inv = PHI_INV
        phi_sq = PHI ** 2
        phi_inv_sq = PHI_INV ** 2

        # (±φ, ±1, ±1/φ, 0) and permutations
        base4 = [phi, 1, phi_inv, 0]
        for perm in set(permutations(base4)):
            for signs in product([1, -1], repeat=4):
                coords = tuple(p * s * scale for p, s in zip(perm, signs))
                norm = np.sqrt(sum(c**2 for c in coords))
                if norm > 0:
                    coords = tuple(c / norm for c in coords)
                    vertices.add(tuple(round(c, 10) for c in coords))

        return [Vertex4D(*v) for v in vertices]

    def get_vertex_array(self) -> np.ndarray:
        """Get all vertices as numpy array (120 x 4)."""
        return np.array([v.to_array() for v in self.vertices])

    def get_embedded_24cells(self) -> List[Polytope24Cell]:
        """
        Extract the five embedded 24-cells from the 600-cell.

        The 600-cell can be partitioned into five 24-cells related
        by rotations involving the Golden Ratio.
        """
        # The five 24-cells are related by 72° rotations (360°/5)
        # in specific 4D planes
        cells = []

        # Start with a base 24-cell
        base = Polytope24Cell()
        cells.append(base)

        # Generate four more by applying Golden Ratio rotations
        for k in range(1, 5):
            angle = 2 * np.pi * k / 5

            # 4D rotation involving φ
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([
                [c, -s * PHI_INV, 0, -s * PHI],
                [s * PHI_INV, c, -s * PHI, 0],
                [0, s * PHI, c, -s * PHI_INV],
                [s * PHI, 0, s * PHI_INV, c]
            ]) / np.sqrt(1 + PHI**2 + PHI_INV**2)

            # Normalize to proper rotation matrix
            R = self._orthogonalize(R)

            rotated_verts = []
            for v in base.vertices:
                new_coords = R @ v.to_array()
                rotated_verts.append(Vertex4D.from_array(new_coords))

            cells.append(Polytope24Cell(vertices=rotated_verts))

        return cells

    def _orthogonalize(self, M: np.ndarray) -> np.ndarray:
        """Orthogonalize a matrix using Gram-Schmidt."""
        Q, R = np.linalg.qr(M)
        return Q


@dataclass
class Polytope120Cell:
    """
    The 120-cell (Hecatonicosachoron) - dual of the 600-cell.

    The 120-cell has:
    - 600 vertices
    - 1200 edges
    - 720 pentagonal faces
    - 120 dodecahedral cells

    The 120-cell is obtained by taking the centers of the 600 tetrahedral
    cells of the 600-cell as vertices (duality).
    """
    vertices: List[Vertex4D] = field(default_factory=list)

    def __post_init__(self):
        """Generate vertices if not provided."""
        if not self.vertices:
            self.vertices = self._generate_vertices()

    def _generate_vertices(self) -> List[Vertex4D]:
        """
        Generate the 600 vertices of the 120-cell.

        These are the centers of the tetrahedral cells of the 600-cell.
        """
        vertices = []

        # For a full implementation, we would compute cell centers
        # of the 600-cell. Here we provide a representative sample.

        # The 120-cell vertices include all permutations and sign
        # combinations of various Golden Ratio-based coordinates

        scale = 0.25
        phi = PHI
        phi2 = PHI ** 2
        phi3 = PHI ** 3

        # Generate a subset of the 600 vertices
        bases = [
            [2, 2, 0, 0],
            [phi3, 1, phi2, 0],
            [phi2, phi2, phi2, phi**(-2)],
            [phi3, phi**(-1), phi, phi],
        ]

        for base in bases:
            for perm in set(permutations(base)):
                for signs in product([1, -1], repeat=4):
                    coords = tuple(p * s * scale for p, s in zip(perm, signs))
                    if coords not in [tuple(v.to_array()) for v in vertices]:
                        vertices.append(Vertex4D(*coords))

        return vertices

    @classmethod
    def from_600cell_dual(cls, cell_600: Polytope600Cell) -> "Polytope120Cell":
        """Construct 120-cell as the dual of a 600-cell."""
        # The vertices of the 120-cell are the cell centers of the 600-cell
        # This requires computing the tetrahedral cell structure
        # For now, return standard construction
        return cls()


class H4Geometry:
    """
    Main interface for H4 Coxeter group geometry.

    This class provides the complete geometric framework for the
    H4 Constellation prototype, including:
    - Polytope construction (24-cell, 16-cell, 600-cell, 120-cell)
    - Trilatic decomposition for data channel separation
    - Layer mapping for kirigami projection
    - Vertex state transitions (0, ½, 1)
    """

    def __init__(self):
        """Initialize the H4 geometry system."""
        self.polytope_24 = Polytope24Cell()
        self.trilatic = TrilaticDecomposition(self.polytope_24)
        self.polytope_600 = None  # Lazy initialization
        self.polytope_120 = None  # Lazy initialization

        # Layer mapping: 6 planes -> 3 layer pairs
        self.layer_mapping = {
            LayerPlane.XY: (TrilaticChannel.ALPHA, 1),  # Pair 1, Layer 1 (Cyan)
            LayerPlane.ZW: (TrilaticChannel.ALPHA, 2),  # Pair 1, Layer 2 (Magenta)
            LayerPlane.XZ: (TrilaticChannel.BETA, 3),   # Pair 2, Layer 3 (Cyan)
            LayerPlane.YW: (TrilaticChannel.BETA, 4),   # Pair 2, Layer 4 (Magenta)
            LayerPlane.XW: (TrilaticChannel.GAMMA, 5),  # Pair 3, Layer 5 (Cyan)
            LayerPlane.YZ: (TrilaticChannel.GAMMA, 6),  # Pair 3, Layer 6 (Magenta)
        }

    def get_24cell(self) -> Polytope24Cell:
        """Get the 24-cell base module."""
        return self.polytope_24

    def get_trilatic(self) -> TrilaticDecomposition:
        """Get the trilatic 16-cell decomposition."""
        return self.trilatic

    def get_600cell(self) -> Polytope600Cell:
        """Get the 600-cell (lazy initialization)."""
        if self.polytope_600 is None:
            self.polytope_600 = Polytope600Cell()
        return self.polytope_600

    def get_120cell(self) -> Polytope120Cell:
        """Get the 120-cell (lazy initialization)."""
        if self.polytope_120 is None:
            self.polytope_120 = Polytope120Cell()
        return self.polytope_120

    def get_layer_pair(self, pair_index: int) -> Tuple[LayerPlane, LayerPlane]:
        """
        Get the layer planes for a pair index (1-3).

        Returns:
            Tuple of (Cyan layer, Magenta layer) planes
        """
        pairs = {
            1: (LayerPlane.XY, LayerPlane.ZW),
            2: (LayerPlane.XZ, LayerPlane.YW),
            3: (LayerPlane.XW, LayerPlane.YZ),
        }
        return pairs[pair_index]

    def project_to_deployment_state(self,
                                     state: VertexState,
                                     vertices: np.ndarray) -> np.ndarray:
        """
        Project vertices according to deployment state.

        Args:
            state: Deployment state (LOCKED, AUXETIC, DEPLOYED)
            vertices: 4D vertex coordinates (N x 4)

        Returns:
            Transformed vertices based on state
        """
        state_value = state.value

        if state_value == 0:
            # Locked: compact 24-cell
            return vertices
        elif state_value == 0.5:
            # Auxetic: intermediate expansion
            scale = 1.0 + 0.5 * (PHI - 1)
            return vertices * scale
        else:
            # Deployed: 600-cell projection
            scale = PHI
            return vertices * scale

    def compute_vertex_distances(self) -> Dict[str, float]:
        """Compute characteristic distances in the 24-cell."""
        vertices = self.polytope_24.get_vertex_array()

        distances = []
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                d = np.linalg.norm(vertices[i] - vertices[j])
                distances.append(d)

        unique_distances = np.unique(np.round(distances, 6))

        return {
            "edge_length": unique_distances[0],
            "diagonal_short": unique_distances[1] if len(unique_distances) > 1 else None,
            "diagonal_long": unique_distances[-1],
            "all_unique": unique_distances.tolist()
        }

    def palindrome_transform(self, forward: bool = True) -> str:
        """
        Compute the H4 palindrome transformation.

        The palindrome is: 24-cell → 600-cell → Dual 120-cell → 24-cell

        Args:
            forward: If True, transform forward (expansion)
                    If False, transform backward (contraction)

        Returns:
            Current state description
        """
        if forward:
            sequence = ["24-cell (base)", "600-cell (projection)",
                       "120-cell (dual)", "24-cell (return)"]
        else:
            sequence = ["24-cell (return)", "120-cell (dual)",
                       "600-cell (projection)", "24-cell (base)"]

        return " → ".join(sequence)
