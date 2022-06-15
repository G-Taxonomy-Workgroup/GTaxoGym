import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.nn.dense import DenseGCNConv, DenseGINConv
from torch_geometric.utils import to_dense_batch


@register_layer('fc_gcnconv')
class FullyConnectedGCNConvLayer(nn.Module):
    """
    Fully connected Graph Convolutional Network (GCN) layer.
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = DenseGCNConv(layer_config.dim_in, layer_config.dim_out,
                                  bias=layer_config.has_bias)

    def forward(self, batch):
        x, mask = to_dense_batch(batch.x, batch.batch)
        # Dense fully connected adjacency matrix for the batch is computed by
        # batched matrix product of the mask: (B x N x 1) @ (B x 1 x N)
        adj = torch.bmm(mask.unsqueeze(2).float(), mask.unsqueeze(1).float()).int()
        x = self.model(x, adj)
        batch.x = x[mask]
        return batch


@register_layer('fc_ginconv')
class FullyConnectedGINConvLayer(nn.Module):
    """
    Fully connected Graph Isomorphism Network (GIN) layer.
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        gin_nn = nn.Sequential(
            Linear_pyg(layer_config.dim_in, layer_config.dim_out), nn.ReLU(),
            Linear_pyg(layer_config.dim_out, layer_config.dim_out))
        self.model = DenseGINConv(gin_nn)

    def forward(self, batch):
        x, mask = to_dense_batch(batch.x, batch.batch)
        # Dense fully connected adjacency matrix for the batch is computed by
        # batched matrix product of the mask: (B x N x 1) @ (B x 1 x N)
        adj = torch.bmm(mask.unsqueeze(2).float(), mask.unsqueeze(1).float())
        x = self.model(x, adj)
        batch.x = x[mask]
        return batch
