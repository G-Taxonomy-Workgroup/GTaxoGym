import logging

import numpy as np
import networkx as nx
import torch
import torch_geometric.utils as pyg_utils
from torch_geometric.graphgym.config import cfg
from torch_geometric.transforms import BaseTransform, ToSparseTensor

from sklearn.cluster import KMeans

from gtaxogym.utils import timeit
from gtaxogym.transform.transforms import pre_transform_in_memory


class BaseSpectralPerturb(BaseTransform):
    """Base class of spectral perturbations.

    Note:
        Do not use this directly, use the derived class instead.
    """

    def __init__(self, **kwargs):
        self._set_params(**kwargs)
        logging.info(self)

    def _set_params(self, **kwargs):
        """Set the perturbation parameters.

        Read (parameter name, value) pairs from kwargs and set them to the
        object. If a passed value for a named parameter is None, then attempt
        to get this parameter value from the global config.

        """
        for kw, val in kwargs.items():
            if val is None:
                cfg_kw = f'{self.__class__.__name__}_{kw}'
                val = getattr(cfg.perturbation, cfg_kw)
            setattr(self, kw, val)

    def __repr__(self):
        var_str = ', '.join(
            f'{name}={val!r}' for name, val in self.__dict__.items()
        )
        return f'{self.__class__.__name__}({var_str})'


class BandpassFiltering(BaseSpectralPerturb):
    """Apply bandpass filtering to the node features over the graph."""

    def __init__(self, band=None):
        """Initialize bandpass filter.

        Args:
            band (str): which band to use ('lo', 'mid', 'hi')
        """
        super().__init__(band=band)

    @property
    def band(self):
        return self._band

    @band.setter
    def band(self, val):
        if val not in ['lo', 'mid', 'hi']:
            raise ValueError(
                f'Unrecognized band option {val!r}, expected lo/mid/hi'
            )
        self._band = val

    @timeit(15, 'BandpassFiltering')
    def __call__(self, data):
        num_nodes = data.num_nodes

        # Prepare the (undirected, unweighted, no self loop) graph Laplacian
        G = pyg_utils.to_networkx(
            data,
            to_undirected=False,  # This conversion option is buggy (PyG 2.0.2); do not use.
            remove_self_loops=True,
        ).to_undirected()
        if not G.edges:  # the graph does not have any edges
            return data
        L = nx.normalized_laplacian_matrix(G, weight=None).toarray()

        # Spectral decomposition.
        logging.detail('   Computing spectral decomposition of '
                       'the normalized graph Laplacian...')
        lams, evects = np.linalg.eigh(L)  # Use eigh to enforce ortho eigvecs.

        total_energy = lams.sum()
        hist, bin_edges = np.histogram(lams, bins=10)
        logging.detail(f'   Total spectrum energy: {total_energy:.2f}\n'
                       f'   Num. zero eigenvalues: {np.sum(lams < 1e-9)}\n'
                       f'   Energy distribution in the spectrum:')
        for i, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            logging.detail(
                f'     num. eigenvals between {start:.2f} and {end:.2f}: '
                f'{hist[i]} ({hist[i] / hist.sum() * 100:.2f}%)'
            )

        logging.detail(f'   Applying bandpass filter: {self.band}')
        evect_slice = self.get_slice(num_nodes)
        selected_idx = lams.argsort()[evect_slice]
        evects = torch.from_numpy(evects[:, selected_idx].astype('float32'))
        lams = lams[selected_idx]

        if lams.size == 0:  # skip filtering if graph too small
            logging.warning(
                f'   The input graph is too small ({num_nodes=}), '
                f'no filtering applied.'
            )
        else: # apply bandpass filtering to the feature matrix.
            x_hat = torch.matmul(evects.T, data.x)
            torch.matmul(evects, x_hat, out=data.x)
            logging.detail(
                f'   Kept {lams.size} eigenvectors with eigenvalues: '
                f'min={lams.min():<6.4f}, max={lams.max():<6.4f}, '
                f'avg={lams.mean():<6.4f} (std={lams.std():<6.4f})\n'
                f'   Kept band energy: {lams.sum():.2f} '
                f'({lams.sum() / total_energy * 100:.2f} %)'
            )

        return data

    def get_slice(self, num_nodes):
        """Determine the slice of eigenpairs to use.

        Split the eigenvectors into three (roughtly) equal sized bin according
        to their corresponding eigenvalues. The bin containing the smallest
        eigenvalues are the 'lo' frequencies; the bin containing the largest
        eigenvalues are the 'hi' frequencies; the remaining are the 'mid'
        frequencies.

        Args:
            num_nodes (int): Number of nodes (maximum number of eigpairs).
        """
        i = ['lo', 'mid', 'hi'].index(self.band)
        num_evect = num_nodes // 3
        start_idx = i * num_evect
        end_idx = start_idx + num_evect if i < 2 else num_nodes
        return slice(start_idx, end_idx)


class WaveletBankFiltering(BaseSpectralPerturb):
    """Apply bandpass filtering to the node features over the graph."""

    def __init__(self, bands=None, norm=None):
        """Initialize bandpass filter.

        Args:
            bands (list): list of booleans (length: num_filters) indicating
                which bands are kept
            norm (str): normalization of the diffusion matrix ('sym' or 'rw')
        """
        super().__init__(bands=bands, norm=norm)

    @timeit(15, 'WaveletBankFiltering')
    def __call__(self, data):
        J = len(self.bands) - 2  # max filter scale
        if J > 6:
            raise ValueError(f"Too much computation.")

        x_prev, x_cur, x_agg = self.init_x(data.x)
        P = self.get_P(data)

        # Iteratively compute Psi_j of x and aggregate relevant responses
        logging.detail('   Aggregat the wavelet responses')
        for j in range(J + 1):
            # x_cur = (P^(2^j)) x, x_prev = (P^(2^(j-1))) x, when j >= 1
            x_cur = self.phi_j_x(P, j, x_cur)
            if self.bands[j]:
                x_agg += x_prev - x_cur
            x_prev[:] = x_cur

        if self.bands[-1]:  # low pass
            x_agg += x_cur

        logging.detail(f'   Original node feature:\n   {data.x}')
        data.x = x_agg.clone().detach().to(data.x.device)
        logging.detail(f'   Filtered node feature:\n   {data.x}')

        return data

    @timeit(15, '   Compute transition matrix P')
    def get_P(self, data):
        """Compute and return the transition matrix.

        The transition matrix P is either in the form of lazy random walk
        matrix 0.5 * (I + M D^{-1}), or the symmetrized version that that,
        i.e. 0.5 * (I + D^{-0.5} M D^{-0.5}).

        Note:
            Ignores self-loops when computing the non-lazy version of the
            transition matrix.

        """
        # Get sparse adjacency matrix
        ToSparseTensor(remove_edge_index=False, fill_cache=False)(data)
        P = data.adj_t.set_diag(0)

        # XXX: not sure whether pyt_utils.degree gives correct ordering
        deg = pyg_utils.degree(
            pyg_utils.remove_self_loops(data.edge_index)[0][0],
            num_nodes=data.num_nodes,
        ).to(P.device())
        deg[deg < 1] = 1  # prevent divide by zero
        invdeg = 1 / deg.unsqueeze(0)
        if self.norm == 'sym':
            sqrtinvdeg = torch.sqrt(invdeg)
            P = P.mul(sqrtinvdeg).t().mul(sqrtinvdeg)
        elif self.norm == 'rw':
            P = P.mul(invdeg)
        else:
            raise ValueError(f"Unknown normalization mode {self.norm!r}")

        P = P.set_diag(1).to_torch_sparse_coo_tensor().to(cfg.device) / 2
        del data.adj_t

        return P

    @staticmethod
    @timeit(15, '   Initialize x')
    def init_x(x):
        """Initialize output and intermediate result placeholders. """
        x_prev = x.clone().detach().to(cfg.device)
        x_cur = x_prev.clone().detach()
        x_agg = torch.zeros(x.size(), device=cfg.device)
        return x_prev, x_cur, x_agg

    @staticmethod
    def phi_j_x(P, j, x):
        """Compute and return (P^{2^j})X.

        If j = -1, simply return PX, which is the same as when j = 0.

        """
        order = 2 ** max(j - 1, 0)
        for _ in range(order):
            x = torch.sparse.mm(P, x)
        return x


class ClusterSparsification(BaseSpectralPerturb):
    """Sparsify and/or fragment inter & intra cluster edges."""

    def __init__(
        self,
        num_evect=None,
        num_cluster=None,
        num_init=None,
        p_inter=None,
        p_intra=None,
        cluster_imb=None,
    ):
        """
        Args:
            num_evect (int): number of utilized eigenvectors / width of
                eigenvector embedding
            num_cluster (int): number of clusters for k-means
            num_init (int): number of time the k-means algorithm will be run
                with different centroid seeds
            p_inter [0,1]: probability of keeping inter-cluster edges
            p_intra [0,1]: probability of keeping intra-cluster edges
            cluster_imb (boolean): use binary encodings of the Fidler vectors
                for cluster assignment if set to true, otherwise use KMeans

        Note:
            The arguments ``num_cluster`` and ``num_init`` are only used when
            ``cluster_imb`` is unset. Otherwise the number of clusters is
            determined by the number of eigenvectors ``num_evect``, which is
            upper bounded by ``2 ** num_evect``.
        """
        super().__init__(
            num_evect=num_evect,
            num_cluster=num_cluster,
            num_init=num_init,
            p_inter=p_inter,
            p_intra=p_intra,
            cluster_imb=cluster_imb,
        )

    @timeit(15, 'ClusterSparsification')
    def __call__(self, data):
        num_evect = self.num_evect      # to be chosen data-driven
        num_cluster = min(self.num_cluster, data.x.shape[0])   # to be chosen data-driven
        num_init = self.num_init        # probably redundant
        p_inter = self.p_inter          # to be chosen data-driven
        p_intra = self.p_intra          # to be chosen data-driven
        cluster_imb = self.cluster_imb

        np.random.seed(cfg.seed)
        num_nodes = data.num_nodes

        # Prepare the (undirected, unweighted, no self loop) graph Laplacian
        G = pyg_utils.to_networkx(
            data,
            to_undirected=False,
            remove_self_loops=True,
        ).to_undirected()
        if not G.edges:  # the graph does not have any edges
            return data
        L = nx.laplacian_matrix(G, weight=None).toarray()
        edgelist_array = np.array(G.edges).T  # undirected edgelist as array
        num_conn_comp = nx.number_connected_components(G)
        
        # Spectral decomposition, only takes non-trivial eigenvectors
        num_evect = min(num_evect, num_nodes - num_conn_comp)
        k = num_evect + num_conn_comp  # offset by number of trivial eigvals
        # NOTE: tried scipy.sparse.linalg.eigsh with 'SM' option, runs even
        # slower than np.linalg.eigh, maybe try 'LM' with shift invert later
        lams, evects = np.linalg.eigh(L)  # use eigh to enforce ortho eigvecs
        selected_idx = lams.argsort()[num_conn_comp : k]
        lams, evects = lams[selected_idx], evects[:, selected_idx]
        logging.detail(
            f'   Initial number of clusters = {num_conn_comp}\n'
            f'   Nontrivial eigenvalues: {lams}'
        )

        if cluster_imb:
            logging.detail(f'   Clustering via binary encodings')
            bin_base = np.power(2, np.arange(num_evect)[::-1])
            encodings = evects > 0
            labels = (encodings * bin_base).sum(axis=1)
        else:
            logging.detail('   Clustering via KMeans')
            kmeans = KMeans(
                n_clusters=num_cluster,
                random_state=cfg.seed,
                n_init=num_init
            ).fit(evects.astype(float))
            labels = kmeans.labels_
        cluster_sizes = [(labels == i).sum() for i in np.unique(labels)]
        logging.detail(f'   Cluster sizes = {sorted(cluster_sizes)[::-1]}')
        
        # Select edges to be deleted (False) and retained (True)
        inter_edge_mask = np.logical_and(
            labels[edgelist_array[0]] != labels[edgelist_array[1]],
            np.random.rand(edgelist_array.shape[1]) > p_inter
        )
        intra_edge_mask = np.logical_and(
            labels[edgelist_array[0]] == labels[edgelist_array[1]],
            np.random.rand(edgelist_array.shape[1]) > p_intra
        )
        edge_mask = ~(inter_edge_mask | intra_edge_mask)
        logging.detail(
            f'   Edges retained: n = {edge_mask.sum()}, percentage ='
            f' {100 * edge_mask.sum() / edge_mask.size:.2f}%\n'
            f'   Edges removed: n = {(~edge_mask).sum()}'
        )
        
        # Prepare the filtered edges (bidirectional)
        new_edgelist_array = edgelist_array[:, edge_mask]
        new_edge_index = np.hstack(
            (new_edgelist_array, new_edgelist_array[[1, 0]])
        )

        data.edge_index = data.edge_index.new_tensor(new_edge_index)

        return data

    def automated_parameters(graph):
        """Automating the hyperparameter search for K-means."""
        raise NotImplementedError

class FiedlerFragmentation(BaseSpectralPerturb):
    """Fragment graph based on Fiedler binary encoding."""

    def __init__(
            self,
            num_iter=None,
            max_size=None,
            method=None,
        ):
        """
        Args:
            num_iter (int): number of splits / new connected components; can
                stop earlier using stopping criterion (below)
            max_size (int): maximal size of connected components; early
                stopping if largest connected component smaller
        """
        super().__init__(num_iter=num_iter, max_size=max_size, method=method)

    @timeit(15, 'FiedlerFragmentation')
    def __call__(self, data):
        num_iter = self.num_iter
        max_size = self.max_size
        method = self.method

        G = pyg_utils.to_networkx(
            data,
            to_undirected=False,
            remove_self_loops=True,
        ).to_undirected()
        if not G.edges:  # the graph does not have any edges
            return data
        
        # derive connected components and order descendingly by size
        subgraphs = list(map(list, nx.connected_components(G)))
        subgraphs.sort(key=len, reverse=False)
        
        logging.detail(
            f'   Initial number of cluster = {len(subgraphs)}\n'
            f'   Initial cluster sizes = {list(map(len, subgraphs[::-1]))}'
        )
        
        for i in range(num_iter):
            subgraph = subgraphs.pop()
            
            # stopping criterion if max. connected comp. size reached
            if len(subgraph) < max_size:
                subgraphs.append(subgraph)
                break
            
            # Preparing the subgraph of the current largest connected component
            S = G.subgraph(subgraph)
            logging.debug(
                f'Calculating Fiedler vector on subgraph with component: '
                f'{list(map(list, nx.connected_components(S)))}'
            )

            # Computing the Fiedler vector (MAJOR BOTTLENECK)
            if method == 'default':
                fiedler = nx.fiedler_vector(S)
            elif method == 'full':
                w, v = np.linalg.eigh(nx.to_numpy_array(S))
                fiedler = v[:, w.argsort()[1]]
            else:
                raise ValueError(
                    f'Unknwon method for computing Fiedler vector {method!r}'
                )
            nodes = np.array(S.nodes)
            
            # split largest comp. in two subgraphs via binary Fiedler enc.
            for new_subgraph in (nodes[fiedler > 0], nodes[fiedler <= 0]):
                # split into smaller comp. if appropriate
                for s in nx.connected_components(G.subgraph(new_subgraph)):
                    subgraphs.append(s)
            
            # order connected components by number of nodes
            subgraphs.sort(key=len)
            
        logging.detail(
            f'   New number of cluster = {len(subgraphs)}\n'
            f'   New cluster sizes = {list(map(len, subgraphs[::-1]))}'
        )
        
        # aggregate edges from subgraphs into single array
        edge_index_list = []
        
        for subgraph in subgraphs:    
            S = G.subgraph(subgraph)
            if S.edges:  # skip if no edges
                edge_index_list.append(np.array(list(map(list, S.edges))).T)
        
        new_edge_index = np.hstack(tuple(edge_index_list))
        
        # transform back to PyG format with "directed" edges in both directions
        new_edge_index = np.hstack((new_edge_index, new_edge_index[[1,0],:]))
        
        logging.detail(
            f'   Original number of edges = {data.edge_index.shape[1]}\n'
            f'   Number of removed of edges = '
            f'{data.edge_index.shape[1]-new_edge_index.shape[1]}'
        )
        
        data.edge_index = data.edge_index.new_tensor(new_edge_index)

        return data
