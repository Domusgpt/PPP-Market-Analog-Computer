"""
Kirigami Sheet - Physical Reservoir for Optical Computing
=========================================================

Implements the full kirigami lattice acting as a Mechanical-Optical
Reservoir Computer as described in Section 7.

Key features:
- Hexagonal (trilatic) arrangement of tristable cells
- Cascading state transitions via mechanical coupling
- Hybrid cut patterns for stiffness programming (attention weights)
- State readout for moiré integration
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Callable
from enum import Enum

from .tristable_cell import TristableCell, CellState, CellProperties


class CutPattern(Enum):
    """
    Hybrid cut patterns from Section 3.2.

    UNIFORM: Equal stiffness everywhere
    RADIAL_SOFT: Soft center, stiff edges (focus attention)
    RADIAL_STIFF: Stiff center, soft edges (peripheral attention)
    GRADIENT_X: Horizontal stiffness gradient
    GRADIENT_Y: Vertical stiffness gradient
    CHECKERBOARD: Alternating stiff/soft regions
    """
    UNIFORM = "uniform"
    RADIAL_SOFT = "radial_soft"
    RADIAL_STIFF = "radial_stiff"
    GRADIENT_X = "gradient_x"
    GRADIENT_Y = "gradient_y"
    CHECKERBOARD = "checkerboard"


@dataclass
class SheetConfig:
    """Configuration for a kirigami sheet."""
    n_cells_x: int = 32           # Cells in x direction
    n_cells_y: int = 32           # Cells in y direction
    lattice_constant: float = 1.0  # Base spacing (micrometers)
    base_stiffness: float = 1.0    # Default cell stiffness
    coupling_strength: float = 0.5  # Inter-cell coupling
    damping: float = 0.1           # Global damping
    cut_pattern: CutPattern = CutPattern.UNIFORM


class KirigamiSheet:
    """
    Kirigami sheet as a physical reservoir computer.

    This class implements the full lattice of tristable cells
    arranged in a hexagonal (trilatic) pattern. It supports:
    - Input injection at specified locations
    - Cascading dynamics via mechanical coupling
    - Stiffness programming via cut patterns
    - State readout as 2D arrays

    From Section 7.1: The kirigami lattice is not a rigid body;
    it is a network of bistable/tristable elastic elements.

    Parameters
    ----------
    config : SheetConfig
        Configuration parameters
    layer_type : str
        "hole_array" (Layer 1, Cyan) or "dot_array" (Layer 2, Red)
    """

    def __init__(
        self,
        config: Optional[SheetConfig] = None,
        layer_type: str = "hole_array"
    ):
        self.config = config or SheetConfig()
        self.layer_type = layer_type

        # Create cell grid
        self.cells: Dict[Tuple[int, int], TristableCell] = {}
        self._initialize_lattice()
        self._setup_neighbors()
        self._apply_cut_pattern()

        # Simulation state
        self.time = 0.0
        self._history: List[np.ndarray] = []

    def _initialize_lattice(self):
        """Create hexagonal lattice of tristable cells."""
        nx, ny = self.config.n_cells_x, self.config.n_cells_y

        for j in range(ny):
            for i in range(nx):
                # Hexagonal offset for odd rows
                offset = 0.5 if j % 2 == 1 else 0.0

                props = CellProperties(
                    stiffness=self.config.base_stiffness,
                    coupling_strength=self.config.coupling_strength,
                    damping=self.config.damping
                )

                cell = TristableCell(
                    position=(i, j),
                    properties=props,
                    initial_state=CellState.CLOSED
                )

                self.cells[(i, j)] = cell

    def _setup_neighbors(self):
        """
        Connect cells to their hexagonal neighbors.

        In a trilatic lattice, each cell has 6 neighbors
        (except at boundaries).
        """
        nx, ny = self.config.n_cells_x, self.config.n_cells_y

        for (i, j), cell in self.cells.items():
            neighbors = []

            # Hexagonal neighbor offsets depend on row parity
            if j % 2 == 0:  # Even row
                neighbor_offsets = [
                    (-1, 0), (1, 0),      # Left, Right
                    (-1, -1), (0, -1),    # Bottom-left, Bottom-right
                    (-1, 1), (0, 1)       # Top-left, Top-right
                ]
            else:  # Odd row
                neighbor_offsets = [
                    (-1, 0), (1, 0),      # Left, Right
                    (0, -1), (1, -1),     # Bottom-left, Bottom-right
                    (0, 1), (1, 1)        # Top-left, Top-right
                ]

            for di, dj in neighbor_offsets:
                ni, nj = i + di, j + dj
                if (ni, nj) in self.cells:
                    neighbors.append(self.cells[(ni, nj)])

            cell.neighbors = neighbors

    def _apply_cut_pattern(self):
        """
        Apply hybrid cut pattern to vary local stiffness.

        From Section 3.2: This spatial variation in stiffness
        acts as the Weight Matrix of the neural network.
        """
        nx, ny = self.config.n_cells_x, self.config.n_cells_y
        cx, cy = nx / 2, ny / 2  # Center coordinates
        max_r = np.sqrt(cx**2 + cy**2)

        pattern = self.config.cut_pattern

        for (i, j), cell in self.cells.items():
            if pattern == CutPattern.UNIFORM:
                stiffness = self.config.base_stiffness

            elif pattern == CutPattern.RADIAL_SOFT:
                # Soft center, stiff edges -> focus attention on center
                r = np.sqrt((i - cx)**2 + (j - cy)**2) / max_r
                stiffness = self.config.base_stiffness * (0.5 + 0.5 * r)

            elif pattern == CutPattern.RADIAL_STIFF:
                # Stiff center, soft edges -> peripheral attention
                r = np.sqrt((i - cx)**2 + (j - cy)**2) / max_r
                stiffness = self.config.base_stiffness * (1.5 - 0.5 * r)

            elif pattern == CutPattern.GRADIENT_X:
                # Horizontal gradient
                stiffness = self.config.base_stiffness * (0.5 + i / nx)

            elif pattern == CutPattern.GRADIENT_Y:
                # Vertical gradient
                stiffness = self.config.base_stiffness * (0.5 + j / ny)

            elif pattern == CutPattern.CHECKERBOARD:
                # Alternating stiff/soft
                if (i + j) % 2 == 0:
                    stiffness = self.config.base_stiffness * 1.5
                else:
                    stiffness = self.config.base_stiffness * 0.5

            cell.properties.stiffness = stiffness

    def inject_input(
        self,
        input_field: np.ndarray,
        input_scale: float = 1.0
    ):
        """
        Inject a 2D input field into the reservoir.

        The input is mapped to cell excitation energies.
        This is how the "visual scene perturbs the reservoir"
        as described in Section 7.1.

        Parameters
        ----------
        input_field : np.ndarray
            2D array of input values (will be resized to match grid)
        input_scale : float
            Scaling factor for input energy
        """
        nx, ny = self.config.n_cells_x, self.config.n_cells_y

        # Resize input to match grid
        from scipy.ndimage import zoom
        if input_field.shape != (ny, nx):
            zoom_y = ny / input_field.shape[0]
            zoom_x = nx / input_field.shape[1]
            input_field = zoom(input_field, (zoom_y, zoom_x), order=1)

        # Normalize input
        input_field = input_field / (np.max(np.abs(input_field)) + 1e-8)

        # Apply to cells
        for (i, j), cell in self.cells.items():
            energy = input_field[j, i] * input_scale
            cell.apply_input(energy)

    def step(self, dt: float = 0.01) -> float:
        """
        Advance simulation by one timestep.

        Implements the "cascading changes" for reservoir computing.
        Cells update based on their internal dynamics and
        neighbor interactions.

        Parameters
        ----------
        dt : float
            Time step

        Returns
        -------
        float
            Total energy released in this step (cascade measure)
        """
        total_energy = 0.0

        # Compute neighbor forces first (before updating)
        neighbor_forces = {}
        for pos, cell in self.cells.items():
            neighbor_forces[pos] = cell.get_neighbor_force()

        # Update all cells
        for pos, cell in self.cells.items():
            energy = cell.update(dt, neighbor_forces[pos])
            total_energy += energy

        self.time += dt
        return total_energy

    def run_cascade(
        self,
        n_steps: int = 100,
        dt: float = 0.01,
        convergence_threshold: float = 1e-4
    ) -> int:
        """
        Run cascading dynamics until convergence or max steps.

        From Section 7.1: When a trigger pushes a cell to switch,
        the strain energy redistributes to neighbors, causing
        a domino cascade of switching events.

        Parameters
        ----------
        n_steps : int
            Maximum number of steps
        dt : float
            Time step
        convergence_threshold : float
            Energy threshold for convergence

        Returns
        -------
        int
            Number of steps taken
        """
        for step in range(n_steps):
            energy = self.step(dt)

            # Record state for history
            self._history.append(self.get_state_field())

            if energy < convergence_threshold:
                return step + 1

        return n_steps

    def get_state_field(self) -> np.ndarray:
        """
        Get current state as 2D array.

        Returns continuous values (0 to 1) for smooth moiré.

        Returns
        -------
        np.ndarray
            2D array of cell values, shape (ny, nx)
        """
        nx, ny = self.config.n_cells_x, self.config.n_cells_y
        field = np.zeros((ny, nx))

        for (i, j), cell in self.cells.items():
            field[j, i] = cell.value

        return field

    def get_discrete_state(self) -> np.ndarray:
        """
        Get current state as discrete values (0, 0.5, 1).

        Returns
        -------
        np.ndarray
            2D array of discrete states
        """
        nx, ny = self.config.n_cells_x, self.config.n_cells_y
        field = np.zeros((ny, nx))

        for (i, j), cell in self.cells.items():
            field[j, i] = cell.state.value

        return field

    def get_transmission_field(self) -> np.ndarray:
        """
        Get optical transmission map.

        This is what the moiré readout layer "sees."

        Returns
        -------
        np.ndarray
            2D transmission coefficients (0 to 1)
        """
        nx, ny = self.config.n_cells_x, self.config.n_cells_y
        field = np.zeros((ny, nx))

        for (i, j), cell in self.cells.items():
            field[j, i] = cell.get_transmission()

        return field

    def set_stiffness_map(self, stiffness_field: np.ndarray):
        """
        Set custom stiffness distribution (weight matrix).

        Parameters
        ----------
        stiffness_field : np.ndarray
            2D array of stiffness values
        """
        nx, ny = self.config.n_cells_x, self.config.n_cells_y

        # Resize if needed
        from scipy.ndimage import zoom
        if stiffness_field.shape != (ny, nx):
            zoom_y = ny / stiffness_field.shape[0]
            zoom_x = nx / stiffness_field.shape[1]
            stiffness_field = zoom(stiffness_field, (zoom_y, zoom_x), order=1)

        for (i, j), cell in self.cells.items():
            cell.properties.stiffness = stiffness_field[j, i]

    def reset(self, state: CellState = CellState.CLOSED):
        """Reset all cells to a uniform state."""
        for cell in self.cells.values():
            cell.set_state(state)

        self.time = 0.0
        self._history.clear()

    def get_statistics(self) -> Dict:
        """
        Get statistical summary of current sheet state.

        Returns
        -------
        Dict
            Statistics including state counts, mean value, etc.
        """
        values = [cell.value for cell in self.cells.values()]
        states = [cell.state for cell in self.cells.values()]

        return {
            "mean_value": np.mean(values),
            "std_value": np.std(values),
            "min_value": np.min(values),
            "max_value": np.max(values),
            "n_closed": sum(1 for s in states if s == CellState.CLOSED),
            "n_intermediate": sum(1 for s in states if s == CellState.INTERMEDIATE),
            "n_open": sum(1 for s in states if s == CellState.OPEN),
            "time": self.time
        }

    def get_history(self) -> np.ndarray:
        """
        Get recorded state history.

        Returns
        -------
        np.ndarray
            3D array (time, ny, nx) of state evolution
        """
        if not self._history:
            return np.array([])
        return np.array(self._history)

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"KirigamiSheet({self.config.n_cells_x}x{self.config.n_cells_y}, "
                f"type={self.layer_type}, mean={stats['mean_value']:.3f})")
