from yacs.config import CfgNode as CN
from torch_geometric.graphgym.register import register_config


@register_config('nx_cfg')
def nx_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.nx = CN()
    # features can be one of ['node_const', 'node_onehot', 'node_clustering_coefficient', 'node_pagerank']
    cfg.nx.augment_feature = ['node_clustering_coefficient', 'node_pagerank']
    cfg.nx.augment_feature_dims = [10, 10]
    cfg.nx.augment_label = 'graph_path_len'
    cfg.nx.augment_label_dims = 5
    cfg.nx.augment_feature_repr = 'original'
