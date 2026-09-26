import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Dense baseline MLP for Fashion-MNIST.

    Architecture: 784 -> 512 -> 512 -> 512 -> 512 -> 512 -> 10 
    The four internal 512x512 weight matrices are used for global compression.
    """

    def __init__(self):
        super().__init__()

        self.linears = nn.ModuleList([
            nn.Linear(784, 512),    # W0
            nn.Linear(512, 512),    # W1  <-- compress
            nn.Linear(512, 512),    # W2  <-- compress
            nn.Linear(512, 512),    # W3  <-- compress
            nn.Linear(512, 512),    # W4  <-- compress
            nn.Linear(512, 10),     # W5
        ])

        self.activation = nn.ReLU()

    def forward(self, x):
        # Fashion-MNIST image : batch x 1 x 28 x 28
        # Flatten : batch x 784 
        x = x.view(x.size(0), -1)

        for layer in self.linears[:-1]:
            x = self.activation(layer(x))

        # Final classification layer
        x = self.linears[-1](x)

        return x


def get_compressible_weights(model):

    # Extract the four 512x512 matrices W1...W4
    weights = []

    for layer_idx in range(1, 5):

        W = model.linears[layer_idx].weight.detach().clone()

        expected_shape = (512, 512)

        if tuple(W.shape) != expected_shape:
            raise ValueError(
                f"Layer {layer_idx} has shape "
                f"{tuple(W.shape)}, expected {expected_shape}"
            )

        weights.append(W)

    return weights


def replace_compressible_weights(model, new_weights):

    if len(new_weights) != 4:
        raise ValueError(
            f"Expected 4 weight matrices, got {len(new_weights)}"
        )

    with torch.no_grad():

        for layer_idx, W in enumerate(new_weights, start=1):

            expected_shape = (512, 512)

            if tuple(W.shape) != expected_shape:
                raise ValueError(
                    f"Replacement weight for layer {layer_idx} "
                    f"has shape {tuple(W.shape)}, "
                    f"expected {expected_shape}"
                )

            target = model.linears[layer_idx].weight

            target.copy_(W.to(device=target.device, dtype=target.dtype )) 