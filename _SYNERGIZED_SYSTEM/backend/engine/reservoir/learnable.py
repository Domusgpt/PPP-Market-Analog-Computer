"""
Learnable Reservoir - Trainable Stiffness Maps
=============================================

Reservoir with learnable attention/stiffness weights.
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass

from ..core.fast_cascade import FastCascadeSimulator


@dataclass
class LearnableResult:
    """Result from learnable reservoir."""
    state: np.ndarray
    stiffness_map: np.ndarray
    loss: Optional[float]


class LearnableReservoir:
    """
    Reservoir with trainable stiffness map.

    The stiffness map acts as learnable attention weights
    that determine how different regions respond to input.

    Training is done via simple gradient-free optimization
    (for compatibility without deep learning frameworks).

    Parameters
    ----------
    grid_size : Tuple[int, int]
        Grid dimensions
    n_basis : int
        Number of basis functions for stiffness
    learning_rate : float
        Update rate for stiffness learning

    Example
    -------
    >>> reservoir = LearnableReservoir((64, 64))
    >>> # Train on examples
    >>> for x, y in training_data:
    ...     reservoir.train_step(x, y, target_fn)
    >>> # Use trained reservoir
    >>> result = reservoir.process(new_input)
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (64, 64),
        n_basis: int = 16,
        learning_rate: float = 0.01
    ):
        self.grid_size = grid_size
        self.n_basis = n_basis
        self.lr = learning_rate

        # Simulator
        self.simulator = FastCascadeSimulator(grid_size)

        # Learnable parameters: basis function weights
        self.weights = np.random.randn(n_basis) * 0.1

        # Create basis functions (Gaussian bumps at different locations)
        self._create_basis()

        # Update stiffness map
        self._update_stiffness()

    def _create_basis(self):
        """Create spatial basis functions."""
        ny, nx = self.grid_size
        y = np.linspace(0, 1, ny)
        x = np.linspace(0, 1, nx)
        Y, X = np.meshgrid(y, x, indexing='ij')

        # Grid of Gaussian centers
        n_side = int(np.sqrt(self.n_basis))
        centers_y = np.linspace(0.1, 0.9, n_side)
        centers_x = np.linspace(0.1, 0.9, n_side)

        self.basis = []
        sigma = 0.2

        for cy in centers_y:
            for cx in centers_x:
                if len(self.basis) >= self.n_basis:
                    break
                basis_fn = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
                self.basis.append(basis_fn)

        # Pad if needed
        while len(self.basis) < self.n_basis:
            self.basis.append(np.ones(self.grid_size) * 0.1)

        self.basis = np.array(self.basis)

    def _update_stiffness(self):
        """Update stiffness map from weights."""
        stiffness = np.zeros(self.grid_size)
        for i, w in enumerate(self.weights):
            stiffness += w * self.basis[i]

        # Ensure positive stiffness
        stiffness = np.abs(stiffness) + 0.1

        # Normalize
        stiffness = stiffness / np.max(stiffness)
        stiffness = 0.5 + 0.5 * stiffness  # Range [0.5, 1.0]

        self.simulator.stiffness = stiffness

    def process(
        self,
        input_field: np.ndarray,
        n_steps: int = 30
    ) -> LearnableResult:
        """
        Process input through reservoir.

        Parameters
        ----------
        input_field : np.ndarray
            Input data
        n_steps : int
            Cascade steps

        Returns
        -------
        LearnableResult
            Processing result
        """
        self.simulator.reset()
        result = self.simulator.run(input_field, n_steps=n_steps)

        return LearnableResult(
            state=result.final_state,
            stiffness_map=self.simulator.stiffness.copy(),
            loss=None
        )

    def train_step(
        self,
        input_field: np.ndarray,
        target: np.ndarray,
        loss_fn: Optional[Callable] = None,
        n_steps: int = 30
    ) -> float:
        """
        Single training step.

        Uses finite difference gradient estimation.

        Parameters
        ----------
        input_field : np.ndarray
            Training input
        target : np.ndarray
            Target output
        loss_fn : Callable, optional
            Loss function (default: MSE)
        n_steps : int
            Cascade steps

        Returns
        -------
        float
            Loss value
        """
        if loss_fn is None:
            loss_fn = lambda pred, tgt: np.mean((pred - tgt) ** 2)

        # Current loss
        result = self.process(input_field, n_steps)
        current_loss = loss_fn(result.state, target)

        # Estimate gradients via finite differences
        epsilon = 0.01
        gradients = np.zeros(self.n_basis)

        for i in range(self.n_basis):
            # Perturb weight
            self.weights[i] += epsilon
            self._update_stiffness()

            # Compute perturbed loss
            result = self.process(input_field, n_steps)
            perturbed_loss = loss_fn(result.state, target)

            # Gradient estimate
            gradients[i] = (perturbed_loss - current_loss) / epsilon

            # Restore weight
            self.weights[i] -= epsilon

        # Update weights
        self.weights -= self.lr * gradients
        self._update_stiffness()

        return current_loss

    def train(
        self,
        inputs: list,
        targets: list,
        n_epochs: int = 10,
        verbose: bool = True
    ) -> list:
        """
        Train on dataset.

        Parameters
        ----------
        inputs : list
            Training inputs
        targets : list
            Target outputs
        n_epochs : int
            Number of epochs
        verbose : bool
            Print progress

        Returns
        -------
        list
            Loss history
        """
        losses = []

        for epoch in range(n_epochs):
            epoch_loss = 0.0

            for x, y in zip(inputs, targets):
                loss = self.train_step(x, y)
                epoch_loss += loss

            epoch_loss /= len(inputs)
            losses.append(epoch_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}")

        return losses

    def get_attention_map(self) -> np.ndarray:
        """
        Get current attention/stiffness map.

        Lower stiffness = higher attention (more responsive).

        Returns
        -------
        np.ndarray
            Attention map (inverted stiffness)
        """
        return 1.0 - self.simulator.stiffness

    def save_weights(self, path: str):
        """Save learned weights."""
        np.save(path, self.weights)

    def load_weights(self, path: str):
        """Load learned weights."""
        self.weights = np.load(path)
        self._update_stiffness()
