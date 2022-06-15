from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


@register_config('perturbation')
def perturbation_cfg(cfg):
    r'''
    This function sets the default config value for perturbation options
    :return: perturbation configuration.
    '''

    cfg.perturbation = CN()

    # Type of perturbation to be applied
    cfg.perturbation.type = 'none'

    # ----------------------------------------------------------------------- #
    # BandpassFiltering options
    # ----------------------------------------------------------------------- #
    # Which band to keep for bandpass filtering (lo, mid, hi)
    cfg.perturbation.BandpassFiltering_band = 'lo'

    # ----------------------------------------------------------------------- #
    # ClusterSparsification options
    # ----------------------------------------------------------------------- #
    # Number of eigenvectors (as embeddings) used for clustering
    cfg.perturbation.ClusterSparsification_num_evect = 4

    # Number of cluster (for k-means, i.e. cluster_imb == False)
    cfg.perturbation.ClusterSparsification_num_cluster = 10

    # Number of time k-means run with different centroid seeds
    cfg.perturbation.ClusterSparsification_num_init = 10

    # Probability of keeping the inter-cluster edges
    cfg.perturbation.ClusterSparsification_p_inter = 0.

    # Probability of keeping the intra-cluster edges
    cfg.perturbation.ClusterSparsification_p_intra = 1.

    # Use binary eigvec encodings (over k-means) for clustering if True
    cfg.perturbation.ClusterSparsification_cluster_imb = True

    # ----------------------------------------------------------------------- #
    # FiedlerFragmentation options
    # ----------------------------------------------------------------------- #
    # Method for computing the Fiedler vector
    cfg.perturbation.FiedlerFragmentation_method = 'default'

    # Maximum number of iteration for splitting the largest connected component
    cfg.perturbation.FiedlerFragmentation_num_iter = 200

    # Minimum of the maximum size of the connected components
    cfg.perturbation.FiedlerFragmentation_max_size = 10

    # ----------------------------------------------------------------------- #
    # FiedlerFragmentation options
    # ----------------------------------------------------------------------- #
    # Indicator for which frequency bands to preserve
    cfg.perturbation.WaveletBankFiltering_bands = [True, True, True]

    # Normalization type ('sym' or 'rw')
    cfg.perturbation.WaveletBankFiltering_norm = 'sym'
