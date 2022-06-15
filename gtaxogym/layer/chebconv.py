import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn.conv.cheb_conv import ChebConv


@register_layer('chebconv')
class ChebConvLayer(nn.Module):
    """ChebConv layer"""

    def __init__(self, layer_config: LayerConfig, **kwargs):
        super(ChebConvLayer, self).__init__()
        self.model = ChebConv(layer_config.dim_in,
                              layer_config.dim_out,
                              K=cfg.gnn.cheb_K,
                              bias=layer_config.has_bias,
                              **kwargs)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch
