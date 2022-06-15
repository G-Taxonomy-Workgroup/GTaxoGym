import logging
import os.path as osp
from functools import partial

import torch_geometric.transforms as T
from torch_geometric.datasets import (Actor, Amazon, CitationFull, Coauthor,
                                      DeezerEurope, FacebookPagePage, Flickr,
                                      GemsecDeezer, GitHub, GNNBenchmarkDataset,
                                      LastFMAsia, Planetoid, Reddit2, TUDataset,
                                      Twitch, WikiCS, WebKB, WikipediaNetwork,
                                      Yelp)
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
        # GraphGym default loader for OGB formatted data.
        dataset = load_ogb(name.replace('_', '-'), dataset_dir)

        if name.startswith('ogbg-'):
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
    if name in ['MNIST', 'CIFAR10', 'PATTERN', 'CLUSTER']:
        if cfg.dataset.split_mode != 'standard':
            raise ValueError(
                "Only 'standard' splits are supported for GNNBenchmarkDataset")
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')
    elif name.startswith('ogbg-'):
        if cfg.dataset.split_mode != 'standard':
            raise ValueError(
                "Only 'standard' splits are supported for OGB datasets")
        set_dataset_splits(dataset, ogbg_splits)
    # Verify or generate dataset train/val/test splits.
    prepare_splits(dataset)
    # T.ToUndirected(dataset)  # convert all graphs to be undirected

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
