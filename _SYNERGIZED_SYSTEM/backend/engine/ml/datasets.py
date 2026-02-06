"""
Dataset Utilities - Data Loading for ML
======================================

Dataset classes for training and evaluation.
"""

import numpy as np
from typing import Optional, Callable, Tuple, List
from pathlib import Path

# Check for PyTorch
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    class MoireDataset(Dataset):
        """
        PyTorch Dataset for moiré-encoded data.

        Encodes raw data on-the-fly or loads pre-encoded patterns.

        Parameters
        ----------
        data : np.ndarray
            Raw data array (N, H, W) or (N, C, H, W)
        labels : np.ndarray
            Labels (N,)
        encoder : object, optional
            Encoder instance for on-the-fly encoding
        transform : callable, optional
            Transform function for data augmentation
        pre_encoded : bool
            If True, data is already encoded patterns

        Example
        -------
        >>> dataset = MoireDataset(images, labels, encoder=encoder)
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> for patterns, labels in loader:
        ...     outputs = model(patterns)
        """

        def __init__(
            self,
            data: np.ndarray,
            labels: np.ndarray,
            encoder=None,
            transform: Optional[Callable] = None,
            pre_encoded: bool = False
        ):
            self.data = data
            self.labels = labels
            self.encoder = encoder
            self.transform = transform
            self.pre_encoded = pre_encoded

            # Cache for encoded patterns
            self._cache = {}

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            # Get raw data
            x = self.data[idx]
            y = self.labels[idx]

            if self.pre_encoded:
                # Already encoded
                pattern = x
            elif idx in self._cache:
                # Use cached
                pattern = self._cache[idx]
            elif self.encoder is not None:
                # Encode on-the-fly
                result = self.encoder.encode_data(x)
                pattern = result['moire_intensity']
                self._cache[idx] = pattern
            else:
                pattern = x

            # Apply transform
            if self.transform is not None:
                pattern = self.transform(pattern)

            # Convert to tensor
            if isinstance(pattern, np.ndarray):
                pattern = torch.from_numpy(pattern).float()
            if pattern.dim() == 2:
                pattern = pattern.unsqueeze(0)  # Add channel dim

            label = torch.tensor(y).long()

            return pattern, label

        def clear_cache(self):
            """Clear encoding cache."""
            self._cache.clear()

        def precompute_all(self):
            """Pre-compute all encodings."""
            if self.encoder is None or self.pre_encoded:
                return

            for i in range(len(self)):
                if i not in self._cache:
                    result = self.encoder.encode_data(self.data[i])
                    self._cache[i] = result['moire_intensity']


    class MoireFeatureDataset(Dataset):
        """
        Dataset that returns feature vectors instead of patterns.

        Parameters
        ----------
        data : np.ndarray
            Raw data (N, H, W)
        labels : np.ndarray
            Labels (N,)
        encoder : object
            Encoder with feature extraction
        """

        def __init__(
            self,
            data: np.ndarray,
            labels: np.ndarray,
            encoder
        ):
            self.data = data
            self.labels = labels
            self.encoder = encoder
            self._cache = {}

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            if idx in self._cache:
                features = self._cache[idx]
            else:
                result = self.encoder.encode_data(self.data[idx])
                features = result['features']
                self._cache[idx] = features

            return (torch.from_numpy(features).float(),
                    torch.tensor(self.labels[idx]).long())


    def create_dataloader(
        data: np.ndarray,
        labels: np.ndarray,
        encoder=None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:
        """
        Create DataLoader for moiré-encoded data.

        Parameters
        ----------
        data : np.ndarray
            Input data
        labels : np.ndarray
            Labels
        encoder : object, optional
            Encoder instance
        batch_size : int
            Batch size
        shuffle : bool
            Shuffle data
        num_workers : int
            Number of worker processes

        Returns
        -------
        DataLoader
            PyTorch DataLoader
        """
        dataset = MoireDataset(data, labels, encoder, **kwargs)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )


else:
    # Dummy implementations

    class MoireDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not installed")

    class MoireFeatureDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not installed")

    def create_dataloader(*args, **kwargs):
        raise ImportError("PyTorch not installed")


# NumPy-only utilities

def create_train_test_split(
    data: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.

    Parameters
    ----------
    data : np.ndarray
        Input data
    labels : np.ndarray
        Labels
    test_size : float
        Fraction for test set
    random_state : int
        Random seed

    Returns
    -------
    Tuple
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)

    n = len(data)
    indices = np.random.permutation(n)

    n_test = int(n * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return (data[train_idx], data[test_idx],
            labels[train_idx], labels[test_idx])


def encode_dataset(
    data: np.ndarray,
    encoder,
    show_progress: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode entire dataset.

    Parameters
    ----------
    data : np.ndarray
        Raw data (N, H, W)
    encoder : object
        Encoder instance
    show_progress : bool
        Print progress

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Encoded patterns and features
    """
    n = len(data)
    patterns = []
    features = []

    for i, x in enumerate(data):
        result = encoder.encode_data(x)
        patterns.append(result['moire_intensity'])
        features.append(result['features'])

        if show_progress and (i + 1) % 100 == 0:
            print(f"Encoded {i+1}/{n}")

    return np.array(patterns), np.array(features)
