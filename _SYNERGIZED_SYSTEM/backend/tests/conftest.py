"""Shared test fixtures for the HEMOC-SGF engine test suite."""

import pytest
import numpy as np


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def small_grid():
    """A small 16x16 grid for fast tests."""
    return (16, 16)


@pytest.fixture
def gradient_input(small_grid):
    """A simple gradient test pattern."""
    nx, ny = small_grid
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    return np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)


@pytest.fixture
def horizontal_stripes(small_grid):
    """Horizontal stripe pattern."""
    nx, ny = small_grid
    y = np.linspace(0, 4 * np.pi, ny)
    return np.sin(y).reshape(-1, 1).repeat(nx, axis=1)


@pytest.fixture
def vertical_stripes(small_grid):
    """Vertical stripe pattern."""
    nx, ny = small_grid
    x = np.linspace(0, 4 * np.pi, nx)
    return np.sin(x).reshape(1, -1).repeat(ny, axis=0)
