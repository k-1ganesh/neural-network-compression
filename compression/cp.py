import numpy as np
import torch
import tensorly as tl
from tensorly.decomposition import parafac
from utils import relative_frobenius_error

tl.set_backend("numpy")

NUM_LAYERS = 4
LAYER_SIZE = 512


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
        raise ValueError(f"Expected {NUM_LAYERS} matrices.")

    tensor = torch.stack(weights,dim=2)

    expected_shape = (LAYER_SIZE,LAYER_SIZE,NUM_LAYERS,)

    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"Unexpected tensor shape: {tuple(tensor.shape)}")

    return tensor


def tensor_to_weights(tensor):
    """
    Convert: 512 x 512 x 4   back to: [W1, W2, W3, W4]
    """

    expected_shape = (LAYER_SIZE,LAYER_SIZE,NUM_LAYERS,)

    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"Unexpected tensor shape: {tuple(tensor.shape)}")

    return [tensor[:, :, i] for i in range(NUM_LAYERS)]


def compress_cp(weights,rank,n_iter_max=300,tol=1e-6,random_state=42):
    """
    Global CP/PARAFAC decomposition.
    Tensor: X in R^(512 x 512 x 4)
    CP approximation: X ~= sum_{r=1}^R a_r o b_r o c_r
    where: A: 512 x R, B: 512 x R, C: 4 x R

    Compressed Parameters Count = 512R + 512R + 4R = 1028R
    """

    if rank < 1:
        raise ValueError(f"Rank must be >= 1, got {rank}")

    tensor = weights_to_tensor(weights)

    # TensorLy expects NumPy arrays with the NumPy backend.
    tensor_np = (tensor.detach().cpu().numpy().astype(np.float64))

    cp_tensor = parafac(
        tensor_np,
        rank=rank,
        n_iter_max=n_iter_max,
        init="random",
        tol=tol,
        random_state=random_state,
        verbose=0,
    )

    tensor_hat_np = tl.cp_to_tensor(cp_tensor)
    tensor_hat_np = np.asarray(tensor_hat_np)
    tensor_hat = torch.from_numpy(tensor_hat_np).to(dtype=tensor.dtype)

    # Reconstruction Error
    reconstruction_error = relative_frobenius_error(tensor, tensor_hat)

    # Split into layers
    reconstructed_weights = tensor_to_weights(tensor_hat)

    # Factors
    factors = cp_tensor.factors
    compressed_parameter_count = sum(factor.size for factor in factors)

    # Convert factors to torch tensors 
    torch_factors = [torch.from_numpy(np.asarray(factor)).to(dtype=tensor.dtype) for factor in factors]

    return {
        "weights": reconstructed_weights,
        "reconstruction_error": reconstruction_error,
        "compressed_parameter_count": compressed_parameter_count,
        "factors": torch_factors,
    }