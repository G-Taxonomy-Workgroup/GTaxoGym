from typing import Optional, Callable, List

import os
import os.path as osp
import pickle

import networkx as nx
import torch
from deepsnap.dataset import GraphDataset
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.graphgym.config import cfg
from torch_geometric.utils.convert import from_networkx

from gtaxogym.transform import feature_augment


class NXDataset(InMemoryDataset):
    names = ['smallworld', 'scalefree', 'ba', 'ba500', 'ws', 'ws500']

    root_url = 'https://github.com/snap-stanford/GraphGym/raw/master/run/datasets'
    urls = {
        'smallworld': f'{root_url}/smallworld.pkl',
        'scalefree': f'{root_url}/scalefree.pkl',
        'ba': f'{root_url}/ba.pkl',
        'ba500': f'{root_url}/ba500.pkl',
        'ws': f'{root_url}/ws.pkl',
        'ws500': f'{root_url}/ws500.pkl',
    }

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        name = self.urls[self.name].split('/')[-1][:-4]
        return [f'{name}.pkl']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        path = download_url(self.urls[self.name], self.raw_dir)
        os.unlink(path)

    def process(self):
        # Read data into huge `Data` list.
        graphs = self.process_nx()
        dataset = GraphDataset(graphs, task=cfg.dataset.task)
        augmentation = feature_augment.FeatureAugment()
        actual_feat_dims, actual_label_dim = augmentation.augment(dataset)
        cfg.nx.augment_feature_dims = actual_feat_dims
        if cfg.nx.augment_label:
            cfg.nx.augment_label_dims = actual_label_dim

        dataset.apply_transform(feature_augment._replace_label,
                                update_graph=True, update_tensor=False)
        data_list = list()
        for g_ds in graphs:
            g_nx = g_ds.G
            g_pyg = from_networkx(g_nx, cfg.nx.augment_feature)
            g_pyg.y = g_nx.graph['graph_label']
            data_list.append(g_pyg)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def process_nx(self) -> List[Data]:
        with open(self.raw_paths[0], 'rb') as f:
            graphs = pickle.load(f)
            if not isinstance(graphs, list):
                graphs = [graphs]

        data_list = []
        for g in graphs:
            if self.pre_filter is not None and not self.pre_filter(g):
                continue
            if self.pre_transform is not None:
                g = self.pre_transform(g)
            data_list.append(g)
        return data_list
