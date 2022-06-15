import logging

import numpy as np
import torch
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils
from numba import njit
from numba.types import bool_
from torch_geometric.graphgym.config import cfg
from torch_geometric.transforms import BaseTransform

from gtaxogym.utils import timeit


def NoFeatures():
    """Replace node features with a 1-d constant vector."""
    return T.Constant(cat=False)


class NodeDegree(BaseTransform):
    """One hot node degree transform.

    For each node, replace its features by a one-hot encoding of its degree.
    If the dataset is composed of only a single graph, i.e. a transductive
    dataset, then positions in a one-hot vector represent *unique* node degree
    values. Else, if the dataset contains multiple graphs, compute the overall
    maximum node degree M and encode degrees in one-hot vectors of length M+1.
    """

    precomputed_max_degree = {
        'COLLAB': 491,
        'IMDB-BINARY': 135,
        'IMDB-MULTI': 88,
        'ENZYMES': 9,
        'PROTEINS': 25,
        'DD': 19,
        'NCI1': 4,
        'PATTERN': 110,
        'CLUSTER': 79,
        'ZINC': 4,
        'MNIST': 8,
        'CIFAR10': 8,
    }

    def __init__(self, dataset, is_transductive):
        if is_transductive:
            self.transform_fn = NodeDegree.one_hot_degree_transductive
        else:
            self.transform_fn = T.OneHotDegree(
                max_degree=NodeDegree.get_max_degree(dataset),
                cat=False
            )

    def __call__(self, data):
        return self.transform_fn(data)

    @staticmethod
    def one_hot_degree_transductive(data):
        deg = pyg_utils.degree(data.edge_index[0])
        unique_sorted_deg = deg.sort().values.unique().numpy()
        deg_map = {j: i for i, j in enumerate(unique_sorted_deg)}
        data.x = torch.zeros((data.num_nodes, unique_sorted_deg.size),
                             device=data.x.device)

        for i, j in enumerate(deg.numpy()):
            data.x[i, deg_map[j]] = 1

        return data

    @staticmethod
    def get_max_degree(dataset):
        if dataset.name in NodeDegree.precomputed_max_degree:
            return NodeDegree.precomputed_max_degree[dataset.name]

        max_degree = 0
        for i in range(len(dataset)):
            try:
                max_degree = max(
                    pyg_utils.degree(dataset[i].edge_index[0]).max().item(),
                    max_degree
                )
            except Exception as e:
                logging.warning(e, dataset[i])
        logging.detail(f"{dataset.name} max node degree: {max_degree}")
        return int(max_degree)


class NoEdges(BaseTransform):
    """Remove all edges from the graph."""

    def __init__(self):
        pass

    def __call__(self, data):
        data.edge_index = torch.tensor([[], []], dtype=torch.long,
                                       device=data.edge_index.device)
        data.edge_attr = None
        return data


class FullyConnected(BaseTransform):
    """Transform graph connectivity into a clique."""

    def __init__(self):
        pass

    def __call__(self, data):
        new_edge_index = list()
        for i in range(data.num_nodes):
            for j in range(data.num_nodes):
                if i != j:
                    new_edge_index.append([i, j])
        new_edge_index = torch.tensor(new_edge_index, dtype=torch.long,
                                      device=data.edge_index.device)
        data.edge_index = new_edge_index.t().contiguous()
        data.edge_attr = None
        return data


class Fragmented(BaseTransform):
    """Fragment to random k-hop neighborhoods."""

    def __init__(self, k):
        """
        Args:
            k (int): propagation size
        """
        self.k = k

    @timeit(15, 'Fragmentation')
    def __call__(self, data):
        num_nodes = data.num_nodes
        k = self.k
        seed = cfg.seed

        # symmetrize and remove self loops
        G = pyg_utils.to_networkx(
            data,
            to_undirected=False,
            remove_self_loops=True,
        ).to_undirected()
        if not G.edges:  # the graph does not have any edges
            return data

        # Construct undirected edge list and make it bidirectional
        edge_index = np.array(G.edges).T
        edge_index = np.hstack([edge_index, edge_index[[1, 0]]])

        if k == 1:
            edge_mask, fragments, seed_node_mask = self.fast_search_k1(
                G.adj, edge_index, num_nodes, seed)
        else:
            edge_mask, fragments, seed_node_mask = self.fast_search(
                edge_index, num_nodes, k, seed)

        # remove inter fragment edges
        data.edge_index = data.edge_index.new_tensor(edge_index[:, edge_mask])

        if cfg.dataset.transductive:
            logging.detail(
                f'   Number of fragments = {seed_node_mask.sum()}\n'
                f'   Edges retained: n = {edge_mask.sum()}, percentage = '
                f'{100 * edge_mask.sum() / edge_mask.size:.2f}%'
            )

        return data

    @staticmethod
    def fast_search_k1(nx_adj, edge_index, num_nodes, seed):
        """One hop fragmentation.

        Special case of k-hop fragmentation. Instead of interating over edges,
        iterate over nodes. And with the help of the adjacency list view of
        nx_adj for one hop breadth first search, the computation can be speed
        up significantly.

        """
        np.random.seed(seed)

        fragment_idx = 0

        fragments = np.zeros(num_nodes, dtype=np.uint32) - np.uint32(1)
        avail_node_mask = np.ones(num_nodes, dtype=bool)
        seed_node_mask = np.zeros(num_nodes, dtype=bool)

        while np.any(avail_node_mask):
            logging.debug(
                f'Number of available nodes = {avail_node_mask.sum()}'
            )

            # randomly draw a seed node from available nodes and record
            seed_node = int(np.random.choice(np.where(avail_node_mask)[0]))
            seed_node_mask[seed_node] = True
            fragments[seed_node] = fragment_idx
            avail_node_mask[seed_node] = False

            # iterate over its one-hop nbhd and mark all available nodes
            one_hop_nbrs = list(nx_adj[seed_node])
            if len(one_hop_nbrs) > 0:
                for nbr in one_hop_nbrs:
                    if avail_node_mask[nbr]:
                        fragments[nbr] = fragment_idx
                        avail_node_mask[nbr] = False

            fragment_idx += 1

        edge_mask = fragments[edge_index[0]] == fragments[edge_index[1]]

        return edge_mask, fragments, seed_node_mask

    @staticmethod
    @njit(nogil=True)
    def fast_search(edge_index, num_nodes, k, seed):
        """Numba accelerated fragmentation.
        This function fragments the graph as GraphFragmenter above. The key
        steps are the following
            - Initialize masks to keep track of available nodes, and unchecked
                edges, etc.
            - Randomly draw seed node from available nodes
            - Perform k-hop propagation by iteratively picking out nodes to be
                considered as neighbors in next iteration, and mark edges as
                checked along the way
            - Mark any edges connecting the "boundary" nodes, i.e. nodes that
                are identified as the neighbors for next iteration, to any
                available node
            - Mark the identified neighbors and next neighbors with the
                corresponding fragment index
            - Repeat until no node is available
            - Pick out intra fragment edges to be retained
        """
        np.random.seed(seed)

        num_edges = edge_index.shape[1]
        fragment_idx = 0

        fragments = np.zeros(num_nodes, dtype=np.uint32) - np.uint32(1)
        seed_node_mask = np.zeros(num_nodes, dtype=bool_)
        avail_node_mask = np.ones(num_nodes, dtype=bool_)
        edge_not_checked = np.ones(num_edges, dtype=bool_)
        edge_not_checked_idx = np.arange(num_edges)

        # masks for current neighbors and neighbors to consider in next step
        curr_nbrs_mask = np.zeros(num_nodes, dtype=bool_)
        next_nbrs_mask = np.zeros(num_nodes, dtype=bool_)

        while np.any(avail_node_mask):
            # Debugging prints
            #print(
            #    f'Number of unchecked edges = {edge_not_checked_idx.size}, '
            #    f'number of available nodes = {avail_node_mask.sum()}'
            #)
            curr_nbrs_mask[:] = next_nbrs_mask[:] = False  # reset neighbor mask

            # randomly draw a seed node from available nodes and record
            seed_node = np.random.choice(np.where(avail_node_mask)[0])
            next_nbrs_mask[seed_node] = seed_node_mask[seed_node] = True

            # perform k-hop propagation
            for _ in range(k + 1):
                # updating neighbors using previously determined neighbors
                curr_nbrs_mask[next_nbrs_mask] = True

                # iterate over unchecked edges and grep those attaching to neighbors
                for i in edge_not_checked_idx:
                    node1, node2 = edge_index[:, i]

                    # pick out edges attaching to any one of current neighbors
                    if curr_nbrs_mask[node1] or curr_nbrs_mask[node2]:
                        next_nbrs_mask[node1] = next_nbrs_mask[node2] = True
                        edge_not_checked[i] = False

                # update unchecked edges
                new_unchecked_ind = edge_not_checked[edge_not_checked_idx]
                edge_not_checked_idx = edge_not_checked_idx[new_unchecked_ind]

            # mark neighbors as unavailable and record fragment index
            avail_node_mask[curr_nbrs_mask] = False
            fragments[curr_nbrs_mask] = fragment_idx

            fragment_idx += 1

        # only retain intra fragment edges
        edge_mask = fragments[edge_index[0]] == fragments[edge_index[1]]

        return edge_mask, fragments, seed_node_mask
