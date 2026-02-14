"""
H4 Kirigami - Six-layer hierarchical kirigami system for H4 Constellation.

This module implements the physical kirigami mechanics for the H4 prototype,
featuring:
- 6 active layers (3 Cyan + 3 Magenta) forming 3 layer pairs
- Tristable states (0, ½, 1) for each layer
- Hierarchical square fractal cut patterns
- Dual-axis actuation per layer
- Moiré interference generation through Cyan/Magenta overlap

The layers correspond to the 6 orthogonal central planes of the 24-cell
projection, enabling physical representation of 4D geometry.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict
import numpy as np


class LayerColor(Enum):
    """Layer color channels for subtractive interference."""
    CYAN = "cyan"       # Transmits Green + Blue, absorbs Red
    MAGENTA = "magenta"  # Transmits Red + Blue, absorbs Green


class DeploymentState(Enum):
    """Kirigami deployment states with physical interpretation."""
    LOCKED = 0.0        # Closed: continuous plate, max stiffness
    AUXETIC = 0.5       # Transition: max auxeticity, bistable point
    DEPLOYED = 1.0      # Open: mesh structure, 600-cell projection


class CutPattern(Enum):
    """Hierarchical cut pattern types."""
    SQUARE_LEVEL1 = "square_L1"      # Basic rotating squares
    SQUARE_LEVEL2 = "square_L2"      # Fractal squares (squares within squares)
    SQUARE_LEVEL3 = "square_L3"      # Deep fractal hierarchy
    CHIRAL_A = "chiral_A"            # Clockwise rotation preference
    CHIRAL_B = "chiral_B"            # Counter-clockwise (mirror of A)


@dataclass
class KirigamiCell:
    """
    A single kirigami unit cell.

    The cell consists of a rotating square element that can transition
    between locked (closed) and deployed (open) states.
    """
    state: float = 0.0  # Continuous state [0, 1]
    rotation_angle: float = 0.0  # Current rotation (radians)
    porosity: float = 0.0  # Open area fraction
    position: Tuple[int, int] = (0, 0)  # Grid position

    # Physical parameters
    hinge_stiffness: float = 1.0  # Normalized stiffness [0, 1]
    max_rotation: float = np.pi / 4  # Maximum rotation (45°)

    def __post_init__(self):
        """Update derived quantities."""
        self._update_from_state()

    def _update_from_state(self):
        """Update rotation and porosity from state."""
        self.rotation_angle = self.state * self.max_rotation
        # Porosity follows a sinusoidal profile
        self.porosity = np.sin(self.rotation_angle) ** 2

    def set_state(self, state: float):
        """Set deployment state and update derived quantities."""
        self.state = np.clip(state, 0.0, 1.0)
        self._update_from_state()

    def get_vertices(self, cell_size: float = 1.0) -> np.ndarray:
        """
        Get the 4 vertices of the rotating square.

        Args:
            cell_size: Size of the unit cell

        Returns:
            4x2 array of vertex coordinates
        """
        cx, cy = self.position
        cx, cy = cx * cell_size, cy * cell_size

        # Square vertices (centered)
        half = cell_size / 2
        base_vertices = np.array([
            [-half, -half],
            [half, -half],
            [half, half],
            [-half, half]
        ])

        # Apply rotation
        c, s = np.cos(self.rotation_angle), np.sin(self.rotation_angle)
        R = np.array([[c, -s], [s, c]])
        rotated = (R @ base_vertices.T).T

        # Translate to position
        return rotated + np.array([cx, cy])

    def get_strain_tensor(self) -> np.ndarray:
        """
        Compute the strain tensor for the current state.

        Returns:
            2x2 strain tensor
        """
        # For a rotating square kirigami:
        # ε_xx = ε_yy = sin(θ) - 1  (auxetic expansion)
        # ε_xy = 0 (for symmetric state)

        strain_val = np.sin(self.rotation_angle) - 1

        return np.array([
            [strain_val, 0],
            [0, strain_val]
        ])


@dataclass
class KirigamiLayer:
    """
    A single kirigami layer in the H4 stack.

    Each layer has:
    - A grid of kirigami cells
    - A color channel (Cyan or Magenta)
    - Dual-axis actuation (longitudinal and transverse)
    - Associated 4D projection plane
    """
    layer_id: int  # 1-6
    color: LayerColor
    grid_size: Tuple[int, int] = (16, 16)
    cells: List[List[KirigamiCell]] = field(default_factory=list)
    cut_pattern: CutPattern = CutPattern.SQUARE_LEVEL2

    # Actuation state
    actuation_longitudinal: float = 0.0  # X-direction actuation
    actuation_transverse: float = 0.0    # Y-direction actuation

    # Physical parameters
    layer_thickness: float = 75e-6  # 75 microns (meters)
    material_transmittance: float = 0.8  # Optical transmittance

    # Associated 4D plane
    plane_axes: Tuple[int, int] = (0, 1)  # Which 4D axes this layer represents

    def __post_init__(self):
        """Initialize cell grid if empty."""
        if not self.cells:
            self._initialize_cells()

    def _initialize_cells(self):
        """Create the grid of kirigami cells."""
        rows, cols = self.grid_size
        self.cells = []

        for i in range(rows):
            row = []
            for j in range(cols):
                # Apply cut pattern - alternate chirality for visual effect
                is_chiral_a = (i + j) % 2 == 0
                cell = KirigamiCell(
                    position=(i, j),
                    hinge_stiffness=self._get_pattern_stiffness(i, j)
                )
                row.append(cell)
            self.cells.append(row)

    def _get_pattern_stiffness(self, i: int, j: int) -> float:
        """Get stiffness based on hierarchical cut pattern."""
        if self.cut_pattern == CutPattern.SQUARE_LEVEL1:
            return 1.0
        elif self.cut_pattern == CutPattern.SQUARE_LEVEL2:
            # Level 2: smaller cuts have higher stiffness
            level = 1 if (i % 2 == 0 and j % 2 == 0) else 0.7
            return level
        elif self.cut_pattern == CutPattern.SQUARE_LEVEL3:
            # Level 3: three scales of stiffness
            if i % 4 == 0 and j % 4 == 0:
                return 1.0
            elif i % 2 == 0 and j % 2 == 0:
                return 0.8
            else:
                return 0.6
        else:
            return 1.0

    def set_state(self, state: float):
        """Set deployment state for all cells."""
        for row in self.cells:
            for cell in row:
                cell.set_state(state)

    def set_actuation(self, longitudinal: float, transverse: float):
        """
        Set dual-axis actuation.

        Args:
            longitudinal: X-direction actuation [0, 1]
            transverse: Y-direction actuation [0, 1]
        """
        self.actuation_longitudinal = np.clip(longitudinal, 0, 1)
        self.actuation_transverse = np.clip(transverse, 0, 1)

        # Compute combined state from dual actuation
        combined_state = (self.actuation_longitudinal +
                         self.actuation_transverse) / 2

        # Apply to cells with gradient based on position
        rows, cols = self.grid_size
        for i, row in enumerate(self.cells):
            for j, cell in enumerate(row):
                # Position-dependent actuation
                x_factor = j / cols * self.actuation_longitudinal
                y_factor = i / rows * self.actuation_transverse
                local_state = combined_state + 0.1 * (x_factor + y_factor)
                cell.set_state(np.clip(local_state, 0, 1))

    def get_state_array(self) -> np.ndarray:
        """Get state values as 2D array."""
        rows, cols = self.grid_size
        states = np.zeros((rows, cols))
        for i, row in enumerate(self.cells):
            for j, cell in enumerate(row):
                states[i, j] = cell.state
        return states

    def get_porosity_array(self) -> np.ndarray:
        """Get porosity values as 2D array."""
        rows, cols = self.grid_size
        porosity = np.zeros((rows, cols))
        for i, row in enumerate(self.cells):
            for j, cell in enumerate(row):
                porosity[i, j] = cell.porosity
        return porosity

    def get_transmittance_map(self) -> np.ndarray:
        """
        Get optical transmittance map for moiré calculation.

        Returns:
            2D array of transmittance values [0, 1]
        """
        porosity = self.get_porosity_array()

        # Transmittance = material transmittance * (1 - porosity) + porosity
        # Open areas have full transmittance, solid areas have material transmittance
        transmittance = (1 - porosity) * self.material_transmittance + porosity

        return transmittance

    def get_color_filter(self) -> np.ndarray:
        """
        Get RGB color filter for this layer.

        Returns:
            3-element array [R, G, B] transmittance
        """
        if self.color == LayerColor.CYAN:
            # Cyan transmits Green and Blue, blocks Red
            return np.array([0.0, 1.0, 1.0])
        else:
            # Magenta transmits Red and Blue, blocks Green
            return np.array([1.0, 0.0, 1.0])

    def discretize_state(self) -> np.ndarray:
        """
        Discretize all cells to 0, 0.5, 1 states.

        Returns:
            2D array of discretized states
        """
        states = self.get_state_array()
        discrete = np.zeros_like(states)

        discrete[states < 0.25] = 0.0
        discrete[(states >= 0.25) & (states < 0.75)] = 0.5
        discrete[states >= 0.75] = 1.0

        return discrete


@dataclass
class LayerPair:
    """
    A pair of complementary Cyan/Magenta layers.

    Each pair controls one of the three trilatic channels
    and is associated with a 16-cell in the decomposition.
    """
    pair_id: int  # 1, 2, or 3
    layer_cyan: KirigamiLayer
    layer_magenta: KirigamiLayer
    trilatic_channel: str  # "alpha", "beta", or "gamma"

    # 4D plane associations
    plane_pair: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 1), (2, 3))

    def set_state(self, state: float):
        """Set both layers to the same state."""
        self.layer_cyan.set_state(state)
        self.layer_magenta.set_state(state)

    def set_differential_state(self, cyan_state: float, magenta_state: float):
        """Set different states for Cyan and Magenta layers."""
        self.layer_cyan.set_state(cyan_state)
        self.layer_magenta.set_state(magenta_state)

    def set_actuation(self,
                       cyan_long: float, cyan_trans: float,
                       magenta_long: float, magenta_trans: float):
        """Set dual-axis actuation for both layers."""
        self.layer_cyan.set_actuation(cyan_long, cyan_trans)
        self.layer_magenta.set_actuation(magenta_long, magenta_trans)

    def compute_moire_intensity(self) -> np.ndarray:
        """
        Compute the moiré intensity pattern from layer overlap.

        The moiré pattern emerges from the interference of the
        two cut patterns when they rotate relative to each other.

        Returns:
            2D intensity array
        """
        # Get transmittance maps
        t_cyan = self.layer_cyan.get_transmittance_map()
        t_magenta = self.layer_magenta.get_transmittance_map()

        # Simple overlap model: multiply transmittances
        overlap = t_cyan * t_magenta

        # Moiré emerges from the difference in rotation angles
        # between corresponding cells
        states_cyan = self.layer_cyan.get_state_array()
        states_magenta = self.layer_magenta.get_state_array()

        # Phase difference creates interference fringes
        phase_diff = np.abs(states_cyan - states_magenta)

        # Moiré intensity modulated by phase difference
        moire = overlap * (1 - 0.5 * np.cos(phase_diff * np.pi))

        return moire

    def compute_spectral_output(self) -> np.ndarray:
        """
        Compute RGB spectral output.

        When Cyan and Magenta overlap:
        - Where both are solid: Only Blue passes
        - Where both are open: White light passes
        - Where only Cyan is open: Cyan (Green + Blue)
        - Where only Magenta is open: Magenta (Red + Blue)

        Returns:
            HxWx3 RGB array
        """
        t_cyan = self.layer_cyan.get_transmittance_map()
        t_magenta = self.layer_magenta.get_transmittance_map()

        rgb_cyan = self.layer_cyan.get_color_filter()
        rgb_magenta = self.layer_magenta.get_color_filter()

        # Compute per-pixel RGB transmittance
        rows, cols = t_cyan.shape
        rgb_output = np.zeros((rows, cols, 3))

        for c in range(3):
            # Combined transmittance for each color channel
            # Both layers must transmit for light to pass
            combined = (1 - t_cyan * (1 - rgb_cyan[c])) * \
                      (1 - t_magenta * (1 - rgb_magenta[c]))
            rgb_output[:, :, c] = combined

        return rgb_output


@dataclass
class H4KirigamiStack:
    """
    The complete 6-layer kirigami stack for H4 Constellation.

    Architecture:
        Layer 1 (Cyan)   ┐
        Layer 2 (Magenta)┘ Pair A (α channel) - XY/ZW planes

        Layer 3 (Cyan)   ┐
        Layer 4 (Magenta)┘ Pair B (β channel) - XZ/YW planes

        Layer 5 (Cyan)   ┐
        Layer 6 (Magenta)┘ Pair C (γ channel) - XW/YZ planes

    This arrangement maximizes moiré interference while maintaining
    the trilatic decomposition mapping to the 24-cell structure.
    """
    grid_size: Tuple[int, int] = (32, 32)
    layers: List[KirigamiLayer] = field(default_factory=list)
    pairs: List[LayerPair] = field(default_factory=list)

    # Inter-layer spacing (for optical calculation)
    layer_spacing: float = 100e-6  # 100 microns

    def __post_init__(self):
        """Initialize the 6-layer stack."""
        if not self.layers:
            self._initialize_stack()

    def _initialize_stack(self):
        """Create the 6 kirigami layers and 3 pairs."""
        # Layer configurations
        configs = [
            (1, LayerColor.CYAN, CutPattern.SQUARE_LEVEL2, (0, 1)),    # XY
            (2, LayerColor.MAGENTA, CutPattern.CHIRAL_B, (2, 3)),       # ZW
            (3, LayerColor.CYAN, CutPattern.SQUARE_LEVEL2, (0, 2)),    # XZ
            (4, LayerColor.MAGENTA, CutPattern.CHIRAL_B, (1, 3)),       # YW
            (5, LayerColor.CYAN, CutPattern.SQUARE_LEVEL2, (0, 3)),    # XW
            (6, LayerColor.MAGENTA, CutPattern.CHIRAL_B, (1, 2)),       # YZ
        ]

        self.layers = []
        for layer_id, color, pattern, plane in configs:
            layer = KirigamiLayer(
                layer_id=layer_id,
                color=color,
                grid_size=self.grid_size,
                cut_pattern=pattern,
                plane_axes=plane
            )
            self.layers.append(layer)

        # Create layer pairs
        self.pairs = [
            LayerPair(
                pair_id=1,
                layer_cyan=self.layers[0],
                layer_magenta=self.layers[1],
                trilatic_channel="alpha",
                plane_pair=((0, 1), (2, 3))
            ),
            LayerPair(
                pair_id=2,
                layer_cyan=self.layers[2],
                layer_magenta=self.layers[3],
                trilatic_channel="beta",
                plane_pair=((0, 2), (1, 3))
            ),
            LayerPair(
                pair_id=3,
                layer_cyan=self.layers[4],
                layer_magenta=self.layers[5],
                trilatic_channel="gamma",
                plane_pair=((0, 3), (1, 2))
            ),
        ]

    def get_layer(self, layer_id: int) -> KirigamiLayer:
        """Get a specific layer by ID (1-6)."""
        return self.layers[layer_id - 1]

    def get_pair(self, pair_id: int) -> LayerPair:
        """Get a specific layer pair by ID (1-3)."""
        return self.pairs[pair_id - 1]

    def set_all_states(self, state: float):
        """Set all layers to the same deployment state."""
        for layer in self.layers:
            layer.set_state(state)

    def set_pair_states(self, alpha: float, beta: float, gamma: float):
        """Set deployment states for each trilatic channel."""
        self.pairs[0].set_state(alpha)
        self.pairs[1].set_state(beta)
        self.pairs[2].set_state(gamma)

    def set_layer_actuations(self, actuations: np.ndarray):
        """
        Set actuation values for all layers.

        Args:
            actuations: Array of shape (6, 2) containing
                       [longitudinal, transverse] for each layer
        """
        for i, layer in enumerate(self.layers):
            layer.set_actuation(actuations[i, 0], actuations[i, 1])

    def apply_quaternion_control(self,
                                  layer_values: np.ndarray):
        """
        Apply quaternion-computed layer values.

        Args:
            layer_values: Array of 6 values from QuaternionController
        """
        for i, layer in enumerate(self.layers):
            layer.set_state(layer_values[i])

    def compute_full_moire(self) -> np.ndarray:
        """
        Compute the full moiré pattern from all layer pairs.

        The three pair moirés are combined to form the complete
        "phason strain" field.

        Returns:
            2D moiré intensity array
        """
        moire_patterns = []

        for pair in self.pairs:
            moire = pair.compute_moire_intensity()
            moire_patterns.append(moire)

        # Combine patterns - multiplicative interference
        combined = moire_patterns[0]
        for moire in moire_patterns[1:]:
            combined = combined * moire

        return combined

    def compute_spectral_moire(self) -> np.ndarray:
        """
        Compute the full RGB spectral moiré output.

        This represents the visible light pattern when the stack
        is backlit with white light.

        Returns:
            HxWx3 RGB array
        """
        # Start with white light
        rows, cols = self.grid_size
        output = np.ones((rows, cols, 3))

        # Apply each layer pair's spectral filtering
        for pair in self.pairs:
            spectral = pair.compute_spectral_output()
            output = output * spectral

        return np.clip(output, 0, 1)

    def get_state_tensor(self) -> np.ndarray:
        """
        Get the complete state tensor for all layers.

        Returns:
            Array of shape (6, H, W) containing all layer states
        """
        states = []
        for layer in self.layers:
            states.append(layer.get_state_array())
        return np.stack(states, axis=0)

    def get_trilatic_states(self) -> Dict[str, float]:
        """
        Get the average state for each trilatic channel.

        Returns:
            Dictionary with alpha, beta, gamma channel states
        """
        return {
            "alpha": np.mean(self.pairs[0].layer_cyan.get_state_array()),
            "beta": np.mean(self.pairs[1].layer_cyan.get_state_array()),
            "gamma": np.mean(self.pairs[2].layer_cyan.get_state_array()),
        }

    def transition_to_state(self,
                             target_state: DeploymentState,
                             steps: int = 10) -> List[np.ndarray]:
        """
        Generate transition sequence to target deployment state.

        Args:
            target_state: Target state (LOCKED, AUXETIC, DEPLOYED)
            steps: Number of interpolation steps

        Returns:
            List of state tensors for animation
        """
        target = target_state.value
        current = np.mean(self.get_state_tensor())

        sequence = []
        for step in range(steps + 1):
            t = step / steps
            intermediate = current + t * (target - current)
            self.set_all_states(intermediate)
            sequence.append(self.get_state_tensor().copy())

        return sequence

    def get_vertex_projection(self) -> np.ndarray:
        """
        Get the effective vertex positions based on current state.

        The kirigami nodes map to 24-cell vertices when deployed.

        Returns:
            Array of 3D projected vertex positions
        """
        states = self.get_trilatic_states()

        # Scale factor based on average deployment
        avg_state = (states["alpha"] + states["beta"] + states["gamma"]) / 3

        # Import geometry for vertex calculation
        from ..geometry.h4_geometry import Polytope24Cell, VertexState

        polytope = Polytope24Cell()
        state_enum = VertexState.DEPLOYED if avg_state > 0.75 else \
                    VertexState.AUXETIC if avg_state > 0.25 else \
                    VertexState.LOCKED

        # Scale vertices based on deployment
        scale = 1.0 + avg_state * 0.618  # Golden ratio expansion

        vertices_3d = polytope.project_to_3d() * scale

        return vertices_3d


class KirigamiMechanics:
    """
    Physics engine for kirigami mechanical simulation.

    Implements the mechanical properties of hierarchical kirigami:
    - Elastic energy calculation
    - Auxetic behavior modeling
    - Bistability at the 0.5 state
    - Neighbor coupling and cascade dynamics
    """

    def __init__(self, stack: H4KirigamiStack):
        """Initialize with a kirigami stack."""
        self.stack = stack

        # Physical constants
        self.elastic_modulus = 2.5e9  # PET film (Pa)
        self.poisson_ratio = 0.33

        # Bistability parameters
        self.bistable_energy_well_depth = 0.1  # Normalized
        self.bistable_transition_width = 0.2

    def compute_elastic_energy(self, layer: KirigamiLayer) -> float:
        """
        Compute total elastic energy stored in a layer.

        Args:
            layer: KirigamiLayer to analyze

        Returns:
            Total elastic energy (normalized)
        """
        energy = 0.0

        for row in layer.cells:
            for cell in row:
                # Hinge bending energy
                hinge_energy = 0.5 * cell.hinge_stiffness * cell.rotation_angle**2

                # Bistability energy contribution
                # Double-well potential at 0, 0.5, 1
                state = cell.state
                well_0 = (state - 0.0)**2
                well_05 = (state - 0.5)**2
                well_1 = (state - 1.0)**2

                bistable_energy = self.bistable_energy_well_depth * min(
                    well_0, well_05, well_1
                )

                energy += hinge_energy + bistable_energy

        return energy

    def compute_poisson_ratio(self, layer: KirigamiLayer) -> float:
        """
        Compute the effective Poisson's ratio of a layer.

        Kirigami structures exhibit auxetic (negative Poisson's ratio)
        behavior near the 0.5 deployment state.

        Returns:
            Effective Poisson's ratio
        """
        avg_state = np.mean(layer.get_state_array())

        # Poisson ratio transitions from positive to negative
        # Minimum (most auxetic) at state = 0.5
        if avg_state < 0.5:
            nu = self.poisson_ratio * (1 - 2 * avg_state)
        else:
            nu = self.poisson_ratio * (2 * avg_state - 1)

        return nu

    def apply_force(self,
                     layer: KirigamiLayer,
                     force_x: float,
                     force_y: float) -> np.ndarray:
        """
        Apply external forces and compute resulting deformation.

        Args:
            layer: Layer to deform
            force_x: Force in x-direction (normalized)
            force_y: Force in y-direction (normalized)

        Returns:
            Displacement field (H, W, 2)
        """
        states = layer.get_state_array()
        rows, cols = states.shape

        # Compute compliance based on porosity
        porosity = layer.get_porosity_array()
        compliance = 1.0 / (1.0 - 0.5 * porosity)  # Higher porosity = more compliant

        # Displacement field
        displacement = np.zeros((rows, cols, 2))

        # Poisson effect
        nu = self.compute_poisson_ratio(layer)

        for i in range(rows):
            for j in range(cols):
                # X displacement
                displacement[i, j, 0] = force_x * compliance[i, j]
                # Y displacement (with Poisson coupling)
                displacement[i, j, 1] = force_y * compliance[i, j] - \
                                       nu * force_x * compliance[i, j]

        return displacement

    def simulate_cascade(self,
                          layer: KirigamiLayer,
                          input_state: float,
                          steps: int = 20) -> List[np.ndarray]:
        """
        Simulate cascade dynamics through the kirigami lattice.

        Args:
            layer: Layer to simulate
            input_state: Initial state applied at boundary
            steps: Number of simulation steps

        Returns:
            List of state arrays over time
        """
        states = layer.get_state_array()
        rows, cols = states.shape

        # Initialize boundary
        states[0, :] = input_state

        history = [states.copy()]

        # Cascade simulation
        for _ in range(steps):
            new_states = states.copy()

            for i in range(1, rows):
                for j in range(cols):
                    # Neighbor influence
                    neighbors = []
                    if i > 0:
                        neighbors.append(states[i-1, j])
                    if i < rows - 1:
                        neighbors.append(states[i+1, j])
                    if j > 0:
                        neighbors.append(states[i, j-1])
                    if j < cols - 1:
                        neighbors.append(states[i, j+1])

                    # Average neighbor state
                    avg_neighbor = np.mean(neighbors)

                    # Update with coupling
                    coupling = 0.2
                    new_states[i, j] = states[i, j] + \
                                      coupling * (avg_neighbor - states[i, j])

            # Apply bistability snapping
            for i in range(rows):
                for j in range(cols):
                    # Snap to nearest stable state
                    if 0.2 < new_states[i, j] < 0.4:
                        new_states[i, j] = 0.3 if np.random.random() > 0.5 else 0.0
                    elif 0.6 < new_states[i, j] < 0.8:
                        new_states[i, j] = 0.7 if np.random.random() > 0.5 else 1.0

            states = new_states
            history.append(states.copy())

            # Update layer
            for i in range(rows):
                for j in range(cols):
                    layer.cells[i][j].set_state(states[i, j])

        return history
