from torch_geometric.graphgym.register import register_config


@register_config('cheb_config')
def cheb_cfg(cfg):
    r'''
    This function sets the default K value for ChebNet convolutions
    :return: perturbation configuration.
    '''
    cfg.gnn.cheb_K = 2


@register_config('gcn2_config')
def gcn2_cfg(cfg):
    r"""
    This function sets the default alpha/theta values for GCN2 convolutions, also adds support for additional residual connections
    :return: perturbation configuration.
    """
    cfg.gnn.alpha = 0.2
    cfg.gnn.theta = None
    cfg.gnn.residual = False
