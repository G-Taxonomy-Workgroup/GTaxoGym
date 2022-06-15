import logging

import torch
import torch.nn.functional as F
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from torch_geometric.graphgym.config import cfg
from tqdm import tqdm


def iterative_pre_transform(data_list, transform_func):
    out_list = list()
    for i in range(len(data_list)):
        out_list.append(transform_func(data_list[i]))
    return out_list


def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    data_list = [transform_func(dataset.get(i))
                 for i in tqdm(range(len(dataset)),
                               disable=not show_progress,
                               miniters=len(dataset) // 20)]
    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)


def typecast_x(data, type_str):
    if type_str == 'float':
        data.x = data.x.float()
    elif type_str == 'long':
        data.x = data.x.long()
    else:
        raise ValueError(f"Unexpected type '{type_str}'.")
    return data


def concat_x_and_pos(data):
    data.x = torch.cat((data.x, data.pos), 1)
    return data


def filter_graphs_by_size(data):
    if data.x.shape[0] < 3500:
        return data
    else:
        logging.info(f'Graph filtered with size: {data.x.shape[0]}')

        
def ogb_molecular_encoding(data):
    """Custom OGB Atom and Edge encoding for molecule dataset.

    The goal is to avoid use of learnable nn.Embedding layers that are used in:
        torch_geometric.graphgym.models.encoder.AtomEncoder
        torch_geometric.graphgym.models.encoder.BondEncoder
    Instead, apply non-learnable one-hot Atom and Bond feature encodings that
    are applicable as a dataset (pre)transformation function.
    """
    encoded_features = []
    for i in range(data.x.shape[1]):
        encoded_features.append(
            F.one_hot(data.x[:, i], num_classes=get_atom_feature_dims()[i])
        )
    data.x = torch.cat(encoded_features, dim=1).float()

    bond_embedding = []
    for i in range(data.edge_attr.shape[1]):
        bond_embedding.append(
            F.one_hot(data.edge_attr[:, i],
                      num_classes=get_bond_feature_dims()[i])
        )
    data.edge_attr = torch.cat(bond_embedding, dim=1).float()

    return data
