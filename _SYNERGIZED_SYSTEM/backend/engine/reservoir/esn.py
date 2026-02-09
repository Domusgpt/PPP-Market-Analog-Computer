"""
Echo State Network Reservoir
===========================

ESN-style formulation of the kirigami reservoir.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ESNResult:
    """Result from ESN processing."""
    states: np.ndarray        # All reservoir states (T, N)
    output: np.ndarray        # Readout output
    final_state: np.ndarray   # Final state


class EchoStateReservoir:
    """
    Echo State Network formulation of the reservoir.

    Classic ESN with grid-based connectivity that mirrors
    the kirigami hexagonal structure.

    Parameters
    ----------
    n_inputs : int
        Input dimension
    n_reservoir : int
        Reservoir size (will be sqrt for 2D grid)
    n_outputs : int
        Output dimension
    spectral_radius : float
        Spectral radius of reservoir weights
    input_scaling : float
        Input weight scaling
    leak_rate : float
        Leaky integration rate

    Example
    -------
    >>> esn = EchoStateReservoir(n_inputs=64, n_reservoir=256)
    >>> esn.fit(X_train, y_train)
    >>> predictions = esn.predict(X_test)
    """

    def __init__(
        self,
        n_inputs: int = 64,
        n_reservoir: int = 256,
        n_outputs: int = 10,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.5,
        leak_rate: float = 0.3
    ):
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate

        # Initialize weights
        self._init_weights()

        # Readout weights (to be trained)
        self.W_out = None

    def _init_weights(self):
        """Initialize reservoir weights with hexagonal connectivity."""
        N = self.n_reservoir
        grid_size = int(np.sqrt(N))

        # Input weights (random)
        self.W_in = (np.random.rand(N, self.n_inputs) - 0.5) * 2 * self.input_scaling

        # Reservoir weights (sparse, hexagonal-like)
        self.W = np.zeros((N, N))

        for i in range(N):
            row = i // grid_size
            col = i % grid_size

            # Connect to hexagonal neighbors
            neighbors = []

            # Direct neighbors
            if col > 0:
                neighbors.append(i - 1)
            if col < grid_size - 1:
                neighbors.append(i + 1)
            if row > 0:
                neighbors.append(i - grid_size)
            if row < grid_size - 1:
                neighbors.append(i + grid_size)

            # Diagonal (hex offset)
            if row > 0 and col > 0:
                neighbors.append(i - grid_size - 1)
            if row < grid_size - 1 and col < grid_size - 1:
                neighbors.append(i + grid_size + 1)

            for j in neighbors:
                if 0 <= j < N:
                    self.W[i, j] = np.random.randn() * 0.5

        # Scale to spectral radius
        eigenvalues = np.linalg.eigvals(self.W)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        if max_eigenvalue > 0:
            self.W *= self.spectral_radius / max_eigenvalue

    def _update_state(self, state: np.ndarray, input_vec: np.ndarray) -> np.ndarray:
        """Update reservoir state."""
        # Leaky integration
        pre_activation = np.dot(self.W_in, input_vec) + np.dot(self.W, state)
        new_state = (1 - self.leak_rate) * state + self.leak_rate * np.tanh(pre_activation)
        return new_state

    def run(
        self,
        inputs: np.ndarray,
        initial_state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Run inputs through reservoir.

        Parameters
        ----------
        inputs : np.ndarray
            Input sequence (T, n_inputs)
        initial_state : np.ndarray, optional
            Initial reservoir state

        Returns
        -------
        np.ndarray
            Reservoir states (T, n_reservoir)
        """
        T = len(inputs)
        states = np.zeros((T, self.n_reservoir))

        if initial_state is None:
            state = np.zeros(self.n_reservoir)
        else:
            state = initial_state.copy()

        for t in range(T):
            state = self._update_state(state, inputs[t])
            states[t] = state

        return states

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        washout: int = 10,
        regularization: float = 1e-6
    ):
        """
        Train readout weights.

        Parameters
        ----------
        inputs : np.ndarray
            Training inputs (T, n_inputs)
        targets : np.ndarray
            Training targets (T, n_outputs)
        washout : int
            Initial steps to discard
        regularization : float
            Ridge regression regularization
        """
        # Run reservoir
        states = self.run(inputs)

        # Discard washout
        states = states[washout:]
        targets = targets[washout:]

        # Ridge regression for readout
        # W_out = (S^T S + lambda I)^-1 S^T Y
        S = states
        Y = targets

        reg_matrix = regularization * np.eye(self.n_reservoir)
        self.W_out = np.linalg.solve(
            np.dot(S.T, S) + reg_matrix,
            np.dot(S.T, Y)
        )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predict outputs for inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Input sequence (T, n_inputs)

        Returns
        -------
        np.ndarray
            Predictions (T, n_outputs)
        """
        if self.W_out is None:
            raise ValueError("Model not trained. Call fit() first.")

        states = self.run(inputs)
        return np.dot(states, self.W_out)

    def process_2d(
        self,
        input_2d: np.ndarray,
        n_steps: int = 10
    ) -> ESNResult:
        """
        Process 2D input (image-like).

        Flattens input and runs through reservoir.

        Parameters
        ----------
        input_2d : np.ndarray
            2D input array
        n_steps : int
            Number of recurrent steps

        Returns
        -------
        ESNResult
            Processing result
        """
        # Flatten and resize input
        flat_input = input_2d.flatten()
        if len(flat_input) != self.n_inputs:
            # Resize
            from scipy.ndimage import zoom
            factor = self.n_inputs / len(flat_input)
            flat_input = zoom(flat_input, factor, order=1)

        # Run for n_steps with same input
        inputs = np.tile(flat_input, (n_steps, 1))
        states = self.run(inputs)

        # Output
        if self.W_out is not None:
            output = np.dot(states[-1], self.W_out)
        else:
            output = states[-1]

        # Reshape final state to 2D
        grid_size = int(np.sqrt(self.n_reservoir))
        final_2d = states[-1].reshape(grid_size, grid_size)

        return ESNResult(
            states=states,
            output=output,
            final_state=final_2d
        )
