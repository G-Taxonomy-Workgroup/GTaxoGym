import logging

from torch_geometric.transforms import Compose
from torch_geometric.graphgym.config import cfg

from gtaxogym.utils import timeit
from gtaxogym.transform.transforms import pre_transform_in_memory
from gtaxogym.transform.perturbations.standard import \
    Fragmented, FullyConnected, NodeDegree, NoEdges, NoFeatures, \
    RandomEdgeRewire, RandomNodeFeatures
from gtaxogym.transform.perturbations.spectral import \
    BandpassFiltering, ClusterSparsification, FiedlerFragmentation, \
    WaveletBankFiltering

__all__ = [
    'ClusterSparsification',
    'FiedlerFragmentation',
    'Fragmented',
    'FullyConnected',
    'NoEdges',
    'NoFeatures',
    'NodeDegree',
    'RandomEdgeRewire',
    'RandomNodeFeatures',
    'WaveletBankFiltering',
]


def perturb_dataset(dataset):
    """Apply a selected perturbation to all graphs in the given dataset.

    Args:
        dataset: PyG dataset object to perturb all the graphs in
    """
    ptype = cfg.perturbation.type
    if ptype == 'none':
        return
    elif ptype.startswith('BandpassFiltering'):
        band = ptype.split('-')[1] if '-' in ptype else None
        fn = BandpassFiltering(band)
    elif ptype == 'ClusterSparsification':
        fn = ClusterSparsification()
    elif ptype == 'FiedlerFragmentation':
        fn = FiedlerFragmentation()
    elif ptype.startswith('Fragmented-k'):
        k = int(ptype.split('-k')[1])
        fn = Fragmented(k=k)
    elif ptype == 'FullyConnected':
        if cfg.gnn.layer_type.startswith('fc_'):
            # When using fully connected conv layers, no edges are needed.
            logging.info("'FullyConnected' achieved by specialized conv layers")
            fn = NoEdges()
        else:
            fn = FullyConnected()
    elif ptype == 'NoEdges':
        fn = NoEdges()
    elif ptype == 'NodeDegree':
        fn = NodeDegree(dataset, cfg.dataset.transductive)
    elif ptype == 'NoFeatures':
        fn = NoFeatures()
    elif ptype == 'NoFeaturesNoEdges':
        fn = Compose([NoEdges(), NoFeatures()])
    elif ptype == 'NoFeaturesFragk1':
        fn = Compose([NoFeatures(), Fragmented(k=1)])
    elif ptype == 'RandomNodeFeatures':
        fn = RandomNodeFeatures(cfg.perturbation.RandomNodeFeatures_feat_dim,
                                cfg.perturbation.RandomNodeFeatures_min_val,
                                cfg.perturbation.RandomNodeFeatures_max_val)
    elif ptype == 'RandomEdgeRewire':
        fn = RandomEdgeRewire(cfg.perturbation.RandomEdgeRewire_rewire_ratio,
                              cfg.perturbation.RandomEdgeRewire_with_replacement)
    elif ptype.startswith('WaveletBankFiltering'):
        bands = None
        if ptype.endswith('-lo'):
            bands = [False, False, True]
        elif ptype.endswith('-mid'):
            bands = [False, True, False]
        elif ptype.endswith('-hi'):
            bands = [True, False, False]
        elif ptype != 'WaveletBankFiltering':
            raise ValueError(f"Unknwon perturbation type: {ptype!r}")
        fn = WaveletBankFiltering(bands=bands)
    else:
        raise ValueError(f"Unexpected perturbation type: '{ptype}'")

    timed_pre_transform_in_memory = timeit(
        logging.INFO,
        'Precomputing perturbations for all graphs...',
        'Perturbations done!'
    )(pre_transform_in_memory)
    timed_pre_transform_in_memory(dataset, fn, show_progress=len(dataset) > 1)
