import numpy as np
import torch
import tensorly as tl
from tensorly.decomposition import tucker
from utils import relative_frobenius_error

# Configuration
NUM_LAYERS = 4
LAYER_SIZE = 512

TENSOR_SHAPE = (LAYER_SIZE,LAYER_SIZE, NUM_LAYERS)

tl.set_backend("numpy")


def weights_to_tensor(weights):
    """
    Convert: [W1, W2, W3, W4]
    into: tensor[:, :, 0] = W1
          tensor[:, :, 1] = W2
          tensor[:, :, 2] = W3
          tensor[:, :, 3] = W4
    Shape: 512 x 512 x 4
    """
    if len(weights) != NUM_LAYERS:
        raise ValueError(f"Expected {NUM_LAYERS} weight matrices, got {len(weights)}.")

    for i, W in enumerate(weights):
        if tuple(W.shape) != (LAYER_SIZE,LAYER_SIZE,):
            raise ValueError(f"W{i + 1} has shape {tuple(W.shape)}. Expected {(LAYER_SIZE, LAYER_SIZE)}.")

    # Stack along mode-3
    tensor = torch.stack(weights,dim=2)

    if tuple(tensor.shape) != TENSOR_SHAPE:
        raise ValueError(f"Unexpected tensor shape {tuple(tensor.shape)}. Expected {TENSOR_SHAPE}.")

    return tensor


def tensor_to_weights(tensor):
    """
    Convert: 512 x 512 x 4   back to: [W1, W2, W3, W4]
    """

    if tuple(tensor.shape) != TENSOR_SHAPE:
        raise ValueError(f"Unexpected tensor shape {tuple(tensor.shape)}. Expected {TENSOR_SHAPE}.")

    weights = [tensor[:, :, i] for i in range(NUM_LAYERS)]

    return weights


def compress_tucker(weights,ranks,n_iter_max=100,tol=1e-6,random_state=42):
    """
    Perform Tucker decomposition on the global weight tensor.

    weights: [W1, W2, W3, W4]
    ranks: [r1, r2, r3]

    Tensor: X ∈ R^(512 × 512 × 4)

    Tucker approximation: X ≈ G ×1 U1 ×2 U2 ×3 U3

    where: G  ∈ R^(r1 × r2 × r3)
           U1 ∈ R^(512 × r1)
           U2 ∈ R^(512 × r2)
           U3 ∈ R^(4 × r3)
    """

    # Validate ranks
    if len(ranks) != 3:
        raise ValueError("Tucker ranks must contain exactly three values: [r1, r2, r3].")

    r1, r2, r3 = ranks

    if r1 < 1 or r1 > LAYER_SIZE:
        raise ValueError(f"Invalid r1={r1}. Must satisfy 1 <= r1 <= {LAYER_SIZE}.")

    if r2 < 1 or r2 > LAYER_SIZE:
        raise ValueError(f"Invalid r2={r2}. Must satisfy 1 <= r2 <= {LAYER_SIZE}.")

    if r3 < 1 or r3 > NUM_LAYERS:
        raise ValueError(f"Invalid r3={r3}. Must satisfy 1 <= r3 <= {NUM_LAYERS}.")

    # Convert weights to tensor
    tensor = weights_to_tensor(weights)

    # TensorLy with NumPy backend
    tensor_np = (tensor.detach().cpu().numpy().astype(np.float64))

    # Tucker decomposition
    core, factors = tucker(
        tensor_np,
        rank=[r1, r2, r3],
        n_iter_max=n_iter_max,
        init="svd",
        tol=tol,
        random_state=random_state,
        verbose=False,
    )

    # Reconstruct tensor
    tensor_hat_np = tl.tucker_to_tensor((core, factors))

    tensor_hat_np = np.asarray(tensor_hat_np)

    tensor_hat = torch.from_numpy(tensor_hat_np).to(dtype=tensor.dtype)

    # Reconstruction error

    reconstruction_error = relative_frobenius_error(tensor, tensor_hat)

    # Convert reconstructed tensor back to four weight matrices
    reconstructed_weights = tensor_to_weights(tensor_hat)

    compressed_parameter_count = (r1*r2*r3 + LAYER_SIZE*r1 + LAYER_SIZE*r2 + NUM_LAYERS*r3)

    torch_factors = [torch.from_numpy(np.asarray(factor)).float() for factor in factors]

    torch_core = torch.from_numpy(np.asarray(core)).float()

    return {
        "weights": reconstructed_weights,
        "reconstruction_error": reconstruction_error,
        "compressed_parameter_count": compressed_parameter_count,
        "core": torch_core,
        "factors": torch_factors,
        "ranks": (r1,r2,r3),
    }