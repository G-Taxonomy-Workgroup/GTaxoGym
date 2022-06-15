import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network


def get_proc_layers(
    dim_in,
    dim_out,
    hidden_channels,
    dropout,
    pre_proc_layer,
    post_proc_layer,
):
    """Get pre and post processing layers.

    Args:
        dim_in (int); input dimension
        dim_out (int): output dimension
        hidden_channels (int): hidden dimmension
        dropout (float): dropout ratio
        pre_proc_layer (bool): if set, use 2 layer MLP for pre-processing
        post_proc_layer (bool): if set, use 2 layer MLP for post-processing

    """
    if pre_proc_layer:  # use 2 layer MLP for pre-processing
        pre_ = torch.nn.Sequential(
            torch.nn.Linear(dim_in, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout) if dropout > 0
            else torch.nn.Identity(),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU()
        )
    else:
        pre_ = torch.nn.Sequential(torch.nn.Identity())

    if post_proc_layer:  # use 2 layer MLP for post-processing
        post_ = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout) if dropout > 0
            else torch.nn.Identity(),
            torch.nn.Linear(hidden_channels, dim_out)
        )
    else:
        post_ = torch.nn.Sequential(torch.nn.Identity())

    return pre_, post_


@register_network('gcn')
class CustomNodeGCN(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        # Retrieve settings from config
        hidden_channels = cfg.gnn.dim_inner
        num_hid_layers = cfg.gnn.layers_mp
        dropout = cfg.gnn.dropout
        pre_proc_layer = True
        post_proc_layer = True

        # Make sure config params are consistent with hard coded params
        if cfg.gnn.layers_pre_mp != 2 or cfg.gnn.layers_post_mp != 2:
            raise ValueError(
                f'The CustomNodeGCN is hard coded with 2 layer MLP as pre- '
                f'and post-processing layers. Modify the config file so that '
                f'cfg.gnn.layeres_pre_mp = cfg.gnn.layers_post_mp = 2.'
            )

        if num_hid_layers < 2:
            raise ValueError(
                f'The number of hidden graph conv layers must be >= 2.'
            )

        # Pre- and post-processing layers
        self.pre_, self.post_ = get_proc_layers(
            dim_in,
            dim_out,
            hidden_channels,
            dropout,
            pre_proc_layer,
            post_proc_layer
        )

        # Determine input and output dimensions of convolution layers
        conv_in = hidden_channels if pre_proc_layer else dim_in
        conv_out = hidden_channels if post_proc_layer else dim_out

        # Convolution with batch normalization
        self.convs = torch.nn.ModuleList()
        self.convs.append(pyg_nn.GCNConv(conv_in, hidden_channels, cached=True))

        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_hid_layers - 2):
            self.convs.append(
                pyg_nn.GCNConv(hidden_channels, hidden_channels, cached=True)
            )
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(pyg_nn.GCNConv(hidden_channels, conv_out, cached=True))

        self.dropout = dropout

    def _apply_index(self, batch):
        mask = '{}_mask'.format(batch.split)
        return batch.x[batch[mask]], batch.y[batch[mask]]

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = self.pre_(x)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        batch.x = self.post_(x)

        pred, label = self._apply_index(batch)

        return pred, label
