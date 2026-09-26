import torch

from utils import relative_frobenius_error


NUM_LAYERS = 4
LAYER_SIZE = 512


def weights_to_matrix(weights):
    """
    Convert: W1, W2, W3, W4   into: M = [W1 W2 W3 W4]   (Via Column wise concatination)
    M.shape: 512 x 2048
    """

    if len(weights) != NUM_LAYERS:
        raise ValueError(f"Expected {NUM_LAYERS} matrices.")

    for i, W in enumerate(weights):
        if tuple(W.shape) != (LAYER_SIZE, LAYER_SIZE):
            raise ValueError(f"W{i + 1} has shape {tuple(W.shape)}")

    matrix = torch.cat(weights, dim=1) # Concatinate all weights column-wise
    expected_shape = (LAYER_SIZE, LAYER_SIZE * NUM_LAYERS)

    if tuple(matrix.shape) != expected_shape:
        raise ValueError(f"Unexpected concatenated shape: {tuple(matrix.shape)}")

    return matrix


def matrix_to_weights(matrix):
    """
    Split: 512 x 2048  back into: W1, W2, W3, W4
    """

    expected_shape = (LAYER_SIZE, LAYER_SIZE * NUM_LAYERS,)

    if tuple(matrix.shape) != expected_shape:
        raise ValueError(f"Unexpected matrix shape: {tuple(matrix.shape)}")

    weights = []

    for i in range(NUM_LAYERS):

        start = i * LAYER_SIZE
        end = (i + 1) * LAYER_SIZE

        W = matrix[:, start:end]
        weights.append(W)

    return weights


def compress_svd(weights, rank):
    """
    Global SVD compression.

    1. Construct: M = [W1 W2 W3 W4]
    2. Compute:   M = U Sigma V^T
    3. Truncate to rank r.
    4. Reconstruct each W_i.

    We absorb Sigma into U: A = U_r Sigma_r      giving: M_hat = A V_r^T
    Storage:
        A:   512 x r
        V^T: r x 2048
    Therefore: Compressed parameters = 2560r
    """

    if rank < 1:
        raise ValueError(f"Rank must be >= 1, got {rank}")

    matrix = weights_to_matrix(weights)

    max_rank = min(matrix.shape)

    if rank > max_rank:
        raise ValueError(f"Rank {rank} exceeds maximum SVD rank {max_rank}.")

    # SVD
    U, S, Vh = torch.linalg.svd(matrix,full_matrices=False,)

    # Truncate

    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    # Absorb Sigma into U.
    A = U_r * S_r.unsqueeze(0)

    # Reconstruct concatenated matrix
    matrix_hat = A @ Vh_r

    # Reconstruction error
    reconstruction_error = relative_frobenius_error(matrix,matrix_hat)

    # Split back into four layers
    reconstructed_weights = matrix_to_weights(matrix_hat)

    # Factorized storage
    compressed_parameter_count = A.numel() + Vh_r.numel()

    return {
        "weights": reconstructed_weights,
        "reconstruction_error": reconstruction_error,
        "compressed_parameter_count": compressed_parameter_count,
        "factors": {"A": A,"Vh": Vh_r,},
    }