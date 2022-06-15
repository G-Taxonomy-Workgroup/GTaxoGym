import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv


@register_layer('gatv2conv')
class GATv2ConvLayer(nn.Module):
    """GATv2 convolution layer
    for forward(), we ignore `edge_attr` since GraphGym does so for GATConv as well.
    Both can be extended to account for `edge_attr` if necessary.
    """

    def __init__(self, layer_config: LayerConfig):
        super(GATv2ConvLayer, self).__init__()
        self.model = GATv2Conv(
            layer_config.dim_in,
            layer_config.dim_out,
            bias=layer_config.has_bias
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch
