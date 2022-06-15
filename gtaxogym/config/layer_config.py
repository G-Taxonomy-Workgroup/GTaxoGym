from torch_geometric.graphgym.register import register_config


@register_config('cheb_config')
def cheb_cfg(cfg):
    r'''
    This function sets the default K value for ChebNet convolutions
    :return: perturbation configuration.
    '''
    cfg.gnn.cheb_K = 2
