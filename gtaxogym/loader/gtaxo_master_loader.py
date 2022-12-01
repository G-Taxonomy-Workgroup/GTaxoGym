import logging
import os.path as osp
from functools import partial
from itertools import chain, accumulate
from typing import List

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import (Actor, Amazon, CitationFull, Coauthor,
                                      DeezerEurope, FacebookPagePage, Flickr,
                                      GemsecDeezer, GitHub, GNNBenchmarkDataset,
                                      LastFMAsia, Planetoid, PPI, Reddit2,
                                      TUDataset, Twitch, WikiCS, WebKB,
                                      WikipediaNetwork, Yelp)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import load_pyg, load_ogb, set_dataset_attr
from torch_geometric.graphgym.register import register_loader

from gtaxogym.loader.malnet_tiny import MalNetTiny
from gtaxogym.loader.nx_dataset import NXDataset
from gtaxogym.loader.split_generator import (prepare_splits,
                                             set_dataset_splits)
from gtaxogym.transform.perturbations import perturb_dataset
from gtaxogym.transform.transforms import (pre_transform_in_memory,
                                           typecast_x, concat_x_and_pos,
                                           filter_graphs_by_size,
                                           ogb_molecular_encoding)


def log_loaded_dataset(dataset, format, name):
    logging.info(f"[*] Loaded dataset '{name}' from '{format}':")
    logging.info(f"  {dataset.data}")
    logging.info(f"  undirected: {dataset[0].is_undirected()}")
    logging.info(f"  num graphs: {len(dataset)}")
    logging.info(f"  avg num_nodes/graph: "
                 f"{int(dataset.data.x.size(0) / len(dataset))}")
    logging.info(f"  avg num_edges/graph: "
                 f"{int(dataset.data.edge_index.size(1) / len(dataset))}")
    logging.info(f"  num node features: {dataset.num_node_features}")
    logging.info(f"  num edge features: {dataset.num_edge_features}")
    if hasattr(dataset, 'num_tasks'):
        logging.info(f"  num tasks: {dataset.num_tasks}")
    logging.info(f"  num classes: {dataset.num_classes}")


def load_pyg_single(dataset_class, dataset_dir, pyg_dataset_id, name):
    """
    Check if the dataset name matches the dataset class name for PyG datasets
    that contain only one dataset. Load and return the dataset.

    Raises:
        ValueError: If the dataset class name differ from the dataset name

    Returns:
        PyG dataset
    """
    if pyg_dataset_id != name != 'none':
        raise ValueError(f"{pyg_dataset_id} class provides only "
                         f"{pyg_dataset_id} dataset, specified name: {name}")
    return dataset_class(dataset_dir)


def quantize_regression_target(dataset, n_bins=20):
    """ Covert regression task to classification by quantile discretization.

    Args:
        dataset: PyG Dataset object
        n_bins: number of quantiles

    Returns:
        Modified dataset object with classification target y.
    """
    from sklearn.preprocessing import KBinsDiscretizer

    train_idx = dataset.split_idxs[0]
    train_ycont = torch.stack([g.y for g in dataset[train_idx]])

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal',
                                   strategy='quantile')
    discretizer.fit(train_ycont)

    all_y = dataset.data.y.unsqueeze(-1).numpy()
    all_y[np.isnan(all_y)] = -1.
    ydisc = discretizer.transform(all_y)
    dataset.data.y = torch.from_numpy(ydisc).long().squeeze()

    # Check.
    # train_ydisc = torch.stack([g.y for g in dataset[train_idx]])
    # print(np.histogram(train_ydisc.numpy(), bins=n_bins))
    # print(np.histogram(dataset.data.y.numpy(), bins=n_bins))

    return dataset


@register_loader('gtaxo-master')
def load_dataset_master(format, name, dataset_dir):
    """
    Master loader that controls loading of all datasets, overshadowing execution
    of any default GraphGym dataset loader. Default GraphGym dataset loader are
    instead called from this function, the format keywords `PyG` and `OGB` are
    reserved for these default GraphGym loaders.

    Custom perturbation transforms and dataset splitting is applied to each
    loaded dataset.

    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset

    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """
    if format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, pyg_dataset_id)

        if pyg_dataset_id == 'Actor':
            dataset = load_pyg_single(Actor, dataset_dir, pyg_dataset_id, name)

        elif pyg_dataset_id == 'Amazon':
            dataset = Amazon(dataset_dir, name)

        elif pyg_dataset_id == 'CitationFull':
            dataset = CitationFull(dataset_dir, name)

        elif pyg_dataset_id == 'Coauthor':
            dataset = Coauthor(dataset_dir, name)

        elif pyg_dataset_id == 'DeezerEurope':
            dataset = load_pyg_single(DeezerEurope, dataset_dir,
                                      pyg_dataset_id, name)

        elif pyg_dataset_id == 'FacebookPagePage':
            dataset = load_pyg_single(FacebookPagePage, dataset_dir,
                                      pyg_dataset_id, name)

        elif pyg_dataset_id == 'Flickr':
            dataset = load_pyg_single(Flickr, dataset_dir, pyg_dataset_id, name)

        elif pyg_dataset_id == 'GemsecDeezer':
            dataset = GemsecDeezer(dataset_dir, name=name,
                                   pre_transform=T.Constant())

        elif pyg_dataset_id == 'GitHub':
            dataset = load_pyg_single(GitHub, dataset_dir, pyg_dataset_id, name)

        elif pyg_dataset_id == 'GNNBenchmarkDataset':
            dataset = preformat_GNNBenchmarkDataset(dataset_dir, name)

        elif pyg_dataset_id == 'LastFMAsia':
            dataset = load_pyg_single(LastFMAsia, dataset_dir, pyg_dataset_id,
                                      name)

        elif pyg_dataset_id == 'MalNetTiny':
            dataset = preformat_MalNetTiny(dataset_dir, feature_set=name)

        elif pyg_dataset_id == 'Planetoid':
            dataset = Planetoid(dataset_dir, name)

        elif pyg_dataset_id == 'PPI':
            dataset = preformat_PPI(dataset_dir)
            dataset.name = 'PPI'

        elif pyg_dataset_id == 'Reddit2':
            dataset = load_pyg_single(Reddit2, dataset_dir, pyg_dataset_id,
                                      name)

        elif pyg_dataset_id == 'TUDataset':
            dataset = preformat_TUDataset(dataset_dir, name)

        elif pyg_dataset_id == 'Twitch':
            dataset = Twitch(dataset_dir, name)

        elif pyg_dataset_id == 'WebKB':
            dataset = WebKB(dataset_dir, name)

        elif pyg_dataset_id == 'WikiCS':
            if pyg_dataset_id != name != 'none':
                raise ValueError(f"WikiCS class provides only the WikiCS "
                                 f"dataset, specified name: {name}")
            dataset = preformat_WikiCS(dataset_dir)

        elif pyg_dataset_id == 'WikipediaNetwork':
            if name == 'crocodile':
                raise NotImplementedError(f"crocodile not implemented yet")
            dataset = WikipediaNetwork(dataset_dir, name)

        elif pyg_dataset_id == 'Yelp':
            dataset = load_pyg_single(Yelp, dataset_dir, pyg_dataset_id, name)

        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

    elif format == 'nx':
        # Custom loader for NetworkX-based datasets from the GraphGym paper.
        dataset_dir = osp.join(dataset_dir, 'nx')
        dataset = NXDataset(dataset_dir, name)

    elif format == 'PyG':
        # GraphGym default loader for Pytorch Geometric datasets.
        dataset = load_pyg(name, dataset_dir)

    elif format == 'OGB':
        if name.startswith('ogbg-'):
            # GraphGym default loader for OGB formatted data.
            dataset = load_ogb(name.replace('_', '-'), dataset_dir)
            # Workaround: Need to set data splits aside because the way they
            # are set in dataset.data breaks iterating over the dataset.
            split_names = [
                'train_graph_index', 'val_graph_index', 'test_graph_index'
            ]
            ogbg_splits = []
            for sn in split_names:
                ogbg_splits.append(dataset.data[sn])
                delattr(dataset.data, sn)

            if name.startswith('ogbg-mol'):
                if (not cfg.dataset.node_encoder) and (not cfg.dataset.edge_encoder):
                    # One-hot encode atoms and bonds as a pre-transform.
                    pre_transform_in_memory(dataset, ogb_molecular_encoding)
                else:
                    logging.warning(
                        'Will use OGB Atom and Bond Encoders, this may not be '
                        'desirable in this project, double check the config!')
        elif name.startswith('PCQM4Mv2-'):
            subset = name.split('-', 1)[1]
            dataset = preformat_OGB_PCQM4Mv2(dataset_dir, subset)
            dataset = quantize_regression_target(dataset, n_bins=cfg.dataset.n_bins)
            dataset.name = name
            ogbg_splits = dataset.split_idxs
            if (not cfg.dataset.node_encoder) and (not cfg.dataset.edge_encoder):
                # One-hot encode atoms and bonds as a pre-transform.
                pre_transform_in_memory(dataset, ogb_molecular_encoding)
            else:
                logging.warning(
                    'Will use OGB Atom and Bond Encoders, this may not be '
                    'desirable in this project, double check the config!')
        else:
            # Special handling of data splits may be needed for other datasets.
            raise ValueError(f"Unsupported OGB dataset: {name}")
    else:
        raise ValueError(f"Unknown data format: {format}")
    log_loaded_dataset(dataset, format, name)

    # Apply perturbation.
    perturb_dataset(dataset)
    logging.info(f"Dataset after perturbation: {cfg.perturbation.type}")
    log_loaded_dataset(dataset, format, name)

    # Set standard dataset train/val/test splits if needed.
    if format in ['PyG-PPI']:
        if cfg.dataset.split_mode != 'standard':
            raise ValueError(
                "Only 'standard' splits are supported for {format} datasets")
        set_dataset_splits(dataset, dataset.split_idxs, dataset.split_names)
        delattr(dataset, 'split_idxs')
        delattr(dataset, 'split_names')
    elif name in ['MNIST', 'CIFAR10', 'PATTERN', 'CLUSTER']:
        if cfg.dataset.split_mode != 'standard':
            raise ValueError(
                "Only 'standard' splits are supported for GNNBenchmarkDataset")
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')
    elif name.startswith('ogbg-') or name.startswith('PCQM4Mv2-'):
        if cfg.dataset.split_mode != 'standard':
            raise ValueError(
                "Only 'standard' splits are supported for OGB datasets")
        set_dataset_splits(dataset, ogbg_splits)
    # Verify or generate dataset train/val/test splits.
    prepare_splits(dataset)
    # T.ToUndirected(dataset)  # convert all graphs to be undirected

    return dataset


def _merge_splits(datasets):
    """Merge a list of PyG dataset into a single dataset.

    Note:
        The original splits are recorded in a newly created :attr:`split_idxs`.

    """
    # TODO: replace merge_splits within preformat_GNNBenchmarkDataset
    def pairwise_range(x):
        # e.g. [1, 3, 4, 7] -> [[1, 2], [3], [4, 5, 6]]
        prev = None
        for i, j in enumerate(x):
            if i > 0:
                yield list(range(prev, j))
            prev = j

    sizes = list(map(len, datasets))
    data_list = list(
        chain.from_iterable(
            map(dataset.get, range(size))
            for dataset, size in zip(datasets, sizes)
        ),
    )  # chain all datasets into a single list
    split_idxs = list(pairwise_range(accumulate(sizes, initial=0)))

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    datasets[0].split_idxs: List[List[int]] = split_idxs

    return datasets[0]


def preformat_PPI(dataset_dir):
    splits = ['train', 'val', 'test']
    dataset = _merge_splits([PPI(root=dataset_dir, split=s) for s in splits])
    dataset.split_names = splits
    return dataset


def preformat_GNNBenchmarkDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's GNNBenchmarkDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """

    def merge_splits():
        splits = ['train', 'val', 'test']
        datasets = [GNNBenchmarkDataset(root=dataset_dir, name=name, split=s)
                    for s in splits]
        n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
        data_list = [datasets[0].get(i) for i in range(n1)] + \
                    [datasets[1].get(i) for i in range(n2)] + \
                    [datasets[2].get(i) for i in range(n3)]

        datasets[0]._indices = None
        datasets[0]._data_list = data_list
        datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
        split_idxs = [list(range(n1)),
                      list(range(n1, n1 + n2)),
                      list(range(n1 + n2, n1 + n2 + n3))]
        datasets[0].split_idxs = split_idxs

        return datasets[0]

    tf_list = []
    if name in ['MNIST', 'CIFAR10']:
        tf_list = [concat_x_and_pos]  # concat pixel value and pos. coordinate
        tf_list.append(partial(typecast_x, type_str='float'))
    else:
        ValueError(f"Loading dataset '{name}' from "
                   f"GNNBenchmarkDataset is not supported.")

    dataset = merge_splits()
    pre_transform_in_memory(dataset, T.Compose(tf_list))

    return dataset


def preformat_MalNetTiny(dataset_dir, feature_set):
    """Load and preformat Tiny version (5k graphs) of MalNet

    Args:
        dataset_dir: path where to store the cached dataset
        feature_set: select what node features to precompute as MalNet
            originally doesn't have any node nor edge features

    Returns:
        PyG dataset object
    """
    if feature_set in ['none', 'Constant']:
        tf = T.Constant()
    elif feature_set == 'OneHotDegree':
        tf = T.OneHotDegree()
    elif feature_set == 'LocalDegreeProfile':
        tf = T.LocalDegreeProfile()
    else:
        raise ValueError(f"Unexpected transform function: {feature_set}")

    dataset = MalNetTiny(dataset_dir)
    dataset.name = 'MalNetTiny'
    logging.info(f'Computing "{feature_set}" node features for MalNetTiny.')
    pre_transform_in_memory(dataset, tf)

    return dataset


def preformat_TUDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's TUDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """
    if name in ['DD', 'NCI1', 'NCI109', 'ENZYMES', 'PROTEINS', 'MUTAG', 'Synthie', 'SYNTHETICnew']:
        func = None
    elif name.startswith('IMDB-') or name.startswith('REDDIT-') \
            or name in ['COLLAB']:
        func = T.Constant()
    else:
        ValueError(f"Loading dataset '{name}' from TUDataset is not supported.")
    if name in ['ENZYMES', 'PROTEINS', 'Synthie', 'SYNTHETICnew']:
        dataset = TUDataset(dataset_dir, name, pre_transform=func, use_node_attr=True)
    else:
        dataset = TUDataset(dataset_dir, name, pre_transform=func, use_node_attr=False)
    if name == 'DD':
        pre_transform_in_memory(dataset, filter_graphs_by_size)
    return dataset


def preformat_WikiCS(dataset_dir):
    """Load and preformat WikiCS dataset.

    WikiCS has 20 train, val, stopping masks but only one test mask, therefore:
    a) combine train and val masks, b) rename stopping mask to val mask,
    c) copy the single test mask 20 times

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object processed to required formatting
    """
    dataset = WikiCS(dataset_dir)

    new_train_mask = dataset.data.train_mask + dataset.data.val_mask
    new_val_mask = dataset.data.stopping_mask
    new_test_mask = dataset.data.test_mask.unsqueeze(1).repeat(
        1,
        dataset.data.train_mask.shape[1]
    )
    set_dataset_attr(dataset, 'train_mask', new_train_mask, len(new_train_mask))
    set_dataset_attr(dataset, 'val_mask', new_val_mask, len(new_val_mask))
    delattr(dataset.data, 'stopping_mask')
    set_dataset_attr(dataset, 'test_mask', new_test_mask, len(new_test_mask))

    return dataset


def preformat_OGB_PCQM4Mv2(dataset_dir, name):
    """Load and preformat PCQM4Mv2 from OGB LSC.
    OGB-LSC provides 4 data index splits:
    2 with labeled molecules: 'train', 'valid' meant for training and dev
    2 unlabeled: 'test-dev', 'test-challenge' for the LSC challenge submission
    We will take random 150k from 'train' and make it a validation set and
    use the original 'valid' as our testing set.
    Note: PygPCQM4Mv2Dataset requires rdkit
    Args:
        dataset_dir: path where to store the cached dataset
        name: select 'subset' or 'full' version of the training set
    Returns:
        PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary.
        from ogb.lsc import PygPCQM4Mv2Dataset
    except Exception as e:
        logging.error('ERROR: Failed to import PygPCQM4Mv2Dataset, '
                      'make sure RDKit is installed.')
        raise e

    import torch
    from numpy.random import default_rng

    dataset = PygPCQM4Mv2Dataset(root=dataset_dir)
    split_idx = dataset.get_idx_split()

    rng = default_rng(seed=42)
    train_idx = rng.permutation(split_idx['train'].numpy())
    train_idx = torch.from_numpy(train_idx)

    # Leave out 150k graphs for a new validation set.
    valid_idx, train_idx = train_idx[:150000], train_idx[150000:]
    if name == 'full':
        split_idxs = [train_idx,  # Subset of original 'train'.
                      valid_idx,  # Subset of original 'train' as validation set.
                      split_idx['valid']  # The original 'valid' as testing set.
                      ]
    elif name == 'subset':
        # Further subset the training set for faster debugging.
        subset_ratio = 0.1
        subtrain_idx = train_idx[:int(subset_ratio * len(train_idx))]
        subvalid_idx = valid_idx[:50000]
        subtest_idx = split_idx['valid']  # The original 'valid' as testing set.
        dataset = dataset[torch.cat([subtrain_idx, subvalid_idx, subtest_idx])]
        n1, n2, n3 = len(subtrain_idx), len(subvalid_idx), len(subtest_idx)
        split_idxs = [list(range(n1)),
                      list(range(n1, n1 + n2)),
                      list(range(n1 + n2, n1 + n2 + n3))]
    else:
        raise ValueError(f'Unexpected OGB PCQM4Mv2 subset choice: {name}')
    dataset.split_idxs = split_idxs
    return dataset
