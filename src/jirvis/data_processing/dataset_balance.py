import numpy as np
from typing import Tuple, Optional
import json
import time


def load_fg_names(fg_key_path: str) -> dict:
    """Load functional group names from json file."""
    with open(fg_key_path, "r") as f:
        fg_keys = json.load(f)
    return {v: k for k, v in fg_keys.items()}


def calculate_ndi(frequencies: np.ndarray) -> float:
    """
    Calculate Nominal Diversity Index (NDI).

    Args:
        frequencies: Array of class frequencies
    Returns:
        float: NDI value between 0 and 1
    """
    n = len(frequencies)
    if n <= 1:
        return 1.0

    # For multi-label data, don't normalize the frequencies to sum to 1
    # since a sample can belong to multiple classes
    ideal_freq = 1.0 / n
    squared_dev = np.sum((frequencies - ideal_freq) ** 2)
    ndi = (1 - squared_dev) / (1 - ideal_freq)

    # Ensure NDI is between 0 and 1
    return max(0.0, min(1.0, ndi))


def optimize_weights(
    ohe: np.ndarray,
    learning_rate: float = 0.01,
    n_iterations: int = 100,
    target_freq: float = None,
) -> np.ndarray:
    """
    Optimize sampling weights using gradient descent.

    Args:
        ohe: One-hot encoded matrix (n_samples, n_features)
        learning_rate: Learning rate for gradient descent
        n_iterations: Number of iterations
        target_freq: Target frequency for each group (if None, use 1/n_features)

    Returns:
        np.ndarray: Optimized weights for each sample
    """
    n_samples, n_features = ohe.shape
    if target_freq is None:
        target_freq = 1.0 / n_features

    # Initialize weights uniformly
    weights = np.ones(n_samples)

    # Precompute molecule signatures (which FGs each molecule has)
    molecule_fgs = [np.where(ohe[i] > 0)[0] for i in range(n_samples)]

    best_weights = weights.copy()
    best_ndi = 0

    for iteration in range(n_iterations):
        # Calculate current frequencies
        weighted_sum = np.sum(ohe * weights[:, np.newaxis], axis=0)
        current_freqs = weighted_sum / np.sum(weights)

        # Calculate NDI for current weights
        current_ndi = calculate_ndi(current_freqs)

        if current_ndi > best_ndi:
            best_ndi = current_ndi
            best_weights = weights.copy()

        # Calculate frequency errors
        freq_errors = current_freqs - target_freq

        # Update weights using gradient descent - vectorized version
        # Multiply OHE matrix by frequency errors to get sum of errors for each molecule
        error_sums = ohe @ freq_errors

        # Count number of functional groups per molecule for averaging
        fg_counts = np.sum(ohe, axis=1)
        fg_counts = np.maximum(fg_counts, 1)  # Avoid division by zero

        # Calculate mean error per molecule
        gradients = error_sums / fg_counts

        # Update weights
        weights -= learning_rate * gradients

        # Ensure positive weights
        weights = np.maximum(weights, 0.1)

        # Normalize weights to maintain stable dataset size
        weights *= (n_samples * 2) / np.sum(weights)

        # Early stopping if converged
        if iteration > 10 and np.abs(gradients).max() < 1e-5:
            break

    return best_weights


def get_dataset_stats(
    multi_label_ohe: np.ndarray, fg_names: Optional[dict] = None
) -> dict:
    """Calculate and return dataset statistics including NDI."""
    # Calculate class frequencies properly for multi-label data
    total_samples = len(multi_label_ohe)
    frequencies = np.sum(multi_label_ohe, axis=0) / total_samples
    ndi = calculate_ndi(frequencies)

    stats = {"ndi": ndi, "frequencies": frequencies, "dataset_size": total_samples}

    if fg_names:
        distribution = {}
        for i, freq in enumerate(frequencies):
            name = fg_names.get(i, f"FG_{i}")
            distribution[name] = freq * 100
        stats["distribution"] = distribution

    print("\nFrequency distribution:")
    for i, freq in enumerate(frequencies):
        print(f"Label {i}: {freq:.3f}")

    return stats


# In dataset_balance.py


def get_balanced_indices_and_ohe(
    train_indices: np.ndarray,
    multi_label_ohe: np.ndarray,
    label_indices: list,
    use_balancing: bool = True,
    use_catch_all: bool = True,
    fg_key_path: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create training indices with optional balancing."""
    # Load FG names if path provided
    fg_names = None
    if fg_key_path:
        try:
            fg_names = load_fg_names(fg_key_path)
        except Exception as e:
            print(f"Warning: Could not load FG names: {e}")

    # Get training subset
    train_ohe = multi_label_ohe[train_indices]

    # Calculate and print initial dataset stats
    stats = get_dataset_stats(train_ohe, fg_names)
    print(f"\nInitial Dataset Statistics:")
    print(f"NDI: {stats['ndi']:.3f}")
    print(f"Dataset size: {stats['dataset_size']}")

    if not use_balancing:
        # Calculate class weights without balancing
        class_frequencies = stats["frequencies"]
        class_weights = 1.0 / (class_frequencies + 1e-5)
        class_weights = class_weights / np.sum(class_weights) * len(class_weights)
        return train_indices, class_weights

    # If balancing is enabled, proceed with optimization
    weights = optimize_weights(train_ohe, learning_rate=0.01, n_iterations=100)

    # Convert weights to repeat counts and create expanded indices
    repeat_counts = np.maximum(1, np.round(weights))
    expanded_indices = np.concatenate(
        [
            np.repeat(idx, int(count))
            for idx, count in zip(train_indices, repeat_counts)
        ],
        dtype=np.int64,
    )

    np.random.shuffle(expanded_indices)

    # Calculate and print balanced dataset stats
    expanded_ohe = multi_label_ohe[expanded_indices]
    balanced_stats = get_dataset_stats(expanded_ohe, fg_names)

    print(f"\nBalanced Dataset Statistics:")
    print(f"NDI: {balanced_stats['ndi']:.3f}")
    print(f"Dataset size: {balanced_stats['dataset_size']}")

    # Calculate class weights for balanced dataset
    class_frequencies = balanced_stats["frequencies"]
    class_weights = 1.0 / (class_frequencies + 1e-5)
    class_weights = class_weights / np.sum(class_weights) * len(class_weights)

    return expanded_indices, class_weights
