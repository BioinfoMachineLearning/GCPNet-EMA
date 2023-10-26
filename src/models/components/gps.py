# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from PyTorch Geometric (https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py):
# -------------------------------------------------------------------------------------------------------------------------------------

import torch
from beartype import beartype
from beartype.typing import Any, Dict, Optional
from torch.nn import BatchNorm1d, Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.nn.attention import PerformerAttention


class RedrawProjection:
    """Helper class to redraw the projection matrices of all fast attention layers."""

    def __init__(self, model: torch.nn.Module, redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        """Redraws the projection matrices of all fast attention layers."""
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules() if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


class GPS(torch.nn.Module):
    """GPS model."""

    def __init__(
        self,
        channels: int,
        pe_walk_length: int,
        pe_dim: int,
        num_edge_channels: int,
        num_layers: int,
        attn_type: str,
        attn_kwargs: Dict[str, Any],
        num_heads: int = 4,
        pool_globally: bool = True,
    ):
        super().__init__()

        self.pool_globally = pool_globally

        self.node_emb = Linear(channels, channels - pe_dim)
        self.pe_lin = Linear(pe_walk_length, pe_dim)
        self.pe_norm = BatchNorm1d(pe_walk_length)
        self.edge_emb = Linear(num_edge_channels, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(
                channels,
                GINEConv(nn),
                heads=num_heads,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs,
            )
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels),
            ReLU(),
            Linear(channels, channels),
            ReLU(),
            Linear(channels, channels),
        )
        self.redraw_projection = RedrawProjection(
            self.convs, redraw_interval=1000 if attn_type == "performer" else None
        )

    @beartype
    def forward(
        self,
        x: torch.Tensor,
        pe: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ):
        """Make a forward pass of the model.

        :param x: Node features of shape [num_nodes, num_node_features].
        :param pe: Positional encoding of shape [num_nodes, num_node_features].
        :param edge_index: Graph connectivity in COO format with shape [2, num_edges].
        :param edge_attr: Edge features of shape [num_edges, num_edge_features].
        :param batch: Batch vector of shape [num_nodes].
        :return: The output of the model.
        """
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        if self.pool_globally:
            x = global_add_pool(x, batch)
        return self.mlp(x)
