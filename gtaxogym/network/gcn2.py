import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNN
from torch_geometric.graphgym.register import register_network


@register_network('custom_gcn2')
class GCN2(GNN):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support additional inputs to the GCN2 layers.
    """

    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__(dim_in, dim_out, **kwargs)

    def forward(self, batch):
        for module in self.children():
            batch.x0 = batch.x  # gcn2conv needs x0 for each layer
            batch = module(batch)
        return batch
