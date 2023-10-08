# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet-EMA (https://github.com/BioinfoMachineLearning/GCPNet-EMA):
# -------------------------------------------------------------------------------------------------------------------------------------

import torch
from beartype import beartype
from beartype.typing import Union
from jaxtyping import Float
from torch.nn import functional as F


@beartype
def _normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalizes a `torch.Tensor` along a given dimension `dim` by its L2 norm.

    From: https://github.com/drorlab/gvp-pytorch

    :param tensor: `torch.Tensor` to normalize.
    :param dim: Dimension along which to normalize.
    :return: Normalized `torch.Tensor`.
    """
    return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


@beartype
def _rbf(
    D: torch.Tensor,
    D_min: float = 0.0,
    D_max: float = 20.0,
    D_count: int = 16,
    device: Union[torch.device, str] = "cpu",
) -> torch.Tensor:
    """Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1. That is, if `D` has
    shape [...dims], then the returned tensor will have shape [...dims, D_count].

    From https://github.com/jingraham/neurips19-graph-protein-design

    :param D: `torch.Tensor` of pairwise distances.
    :param D_min: Minimum distance to consider.
    :param D_max: Maximum distance to consider.
    :param D_count: Number of RBFs to use.
    :param device: Device to use.
    :return: RBF embedding of `D`.
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


def _orientations(
    X: Float[torch.Tensor, "num_nodes 3"]  # noqa: F722
) -> Float[torch.Tensor, "num_nodes 2 3"]:  # noqa: F722
    """Orientations of edges in a graph.

    :param X: `torch.Tensor` of node coordinates.
    :return: `torch.Tensor` of edge orientations.
    """
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat((forward.unsqueeze(-2), backward.unsqueeze(-2)), dim=-2)
