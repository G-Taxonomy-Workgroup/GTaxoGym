import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn.conv.gcn2_conv import GCN2Conv


@register_layer('gcn2conv')
class GCN2ConvLayer(nn.Module):
    """GCN2Conv layer from https://arxiv.org/abs/2007.02133."""

    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.dropout = cfg.gnn.dropout
        self.residual = cfg.gnn.residual

        self.model = GCN2Conv(layer_config.dim_in,
                              cfg.gnn.alpha,
                              cfg.gnn.theta,
                              **kwargs)
        # alpha value is set using results from the GCNII paper

    def forward(self, batch):
        x_in = batch.x
        batch.x = self.model(batch.x, batch.x0, batch.edge_index)
        batch.x = F.relu(batch.x)
        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)

        if self.residual:
            batch.x = x_in + batch.x

        return batch
