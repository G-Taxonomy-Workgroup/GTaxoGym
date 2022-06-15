import argparse
import logging
import os
from sys import path

import numpy as np
import torch

path.append('..')
from plotting import checkdir

from matplotlib import pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx as to_nx
from torch_geometric.graphgym.config import cfg, set_cfg

from gtaxogym.loader.gtaxo_master_loader import load_dataset_master
from gtaxogym.transform.perturbations import BandpassFiltering, \
    ClusterSparsification, FiedlerFragmentation, Fragmented, \
    WaveletBankFiltering, NodeDegree, FullyConnected, NoFeatures, NoEdges

set_cfg(cfg)
PERT_FUNC_DICT = {
    'FullyConnected': FullyConnected(),
    'NodeDegree': None,  # Has to be constructed with a loaded dataset.
    'NoEdges': NoEdges(),
    'NoFeatures': NoFeatures(),
    'ClusterSparsification': ClusterSparsification(),
    'FiedlerFragmentation': FiedlerFragmentation(),
    'Fragmented-k1': Fragmented(k=1),
    'Fragmented-k2': Fragmented(k=2),
    'Fragmented-k3': Fragmented(k=3),
    'BandpassLow': BandpassFiltering('lo'),
    'BandpassMid': BandpassFiltering('mid'),
    'BandpassHigh': BandpassFiltering('hi'),
    'WaveletBankLow': WaveletBankFiltering(bands=[False, False, True]),
    'WaveletBankMid': WaveletBankFiltering(bands=[False, True, False]),
    'WaveletBankHigh': WaveletBankFiltering(bands=[True, False, False]),
}
NUM_PERT = len(PERT_FUNC_DICT)


def parse_args():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--dataset',
        default='PyG-TUDataset',
        help='Dataset to use.',
    )

    parser.add_argument(
        '--name',
        default='ENZYMES',
        help='Specific name of the dataset to use.'
    )

    parser.add_argument(
        '--selected_index',
        type=int,
        nargs='+',
        default=[0, 20, 40, 50],
        help='Selected graphs to be ploted.',
    )

    parser.add_argument(
        '--savefigfile',
        default='outputs/perturbation/enzyme_pert.pdf',
        help='Where to save the output figure file.',
    )

    return parser.parse_args()


def _set_ylabel(ax, ylabel, fontsize):
    # Turn on axis, disabled by nx.draw
    ax.axis('on')

    # Hide the frame (border)
    for which in ['top', 'right', 'bottom', 'left']:
        ax.spines[which].set_visible(False)

    # Set ylabel
    ax.set_ylabel(ylabel, fontsize=fontsize)
    

def plot_graph_comparison(
    graphs,
    savefigfile=None,
    subplot_size=5,
    node_size=80,
    fontsize=16,
    cmap=plt.get_cmap('coolwarm'),
):
    """Plot comparison of graphs with different perturbations.

    For each graph, plot the original graph and also the perturbed graphs
    constructed using the perturbation functions in PERT_FUNC_DICT.

    Args:
        graphs: list of graphs as PyG dataset
        savefigfile (str): where to save to figure, plot directly if None
        subplot_size (int): size of each subplot
        node_size (int): size of the node of the graphs
        fontsize (int): font size of titles and ylabels
        cmat: color map used
    """
    num_graph = len(graphs)
    fig = plt.figure(
        figsize=(subplot_size * (NUM_PERT + 1), subplot_size * num_graph)
    )

    for i, graph in enumerate(graphs):
        nxg = to_nx(graph, to_undirected=True)
        pos = nx.kamada_kawai_layout(nxg)
        offset = (NUM_PERT + 1) * i

        ax = fig.add_subplot(num_graph, NUM_PERT + 1, offset + 1)
        nx.draw(nxg, pos, cmap=cmap, node_size=node_size, ax=ax)
        _set_ylabel(ax, f'ENZYME #{i}', fontsize)

        if i == 0:
            ax.set_title('Original', fontsize=fontsize)

        for j, (pert_name, pert_func) in enumerate(PERT_FUNC_DICT.items()):
            nxg_pert = to_nx(pert_func(graph.clone()), to_undirected=True)

            ax = fig.add_subplot(num_graph, NUM_PERT + 1, offset + j + 2)
            nx.draw(nxg_pert, pos, cmap=cmap, node_size=node_size, ax=ax)

            if i == 0:
                ax.set_title(pert_name, fontsize=fontsize)

    if savefigfile is None:
        plt.tight_layout()
        plt.show()
    else:
        checkdir(os.path.dirname(savefigfile))
        plt.savefig(savefigfile, bbox_inches='tight')


def plot_individual_perturbations(graph,
                                  savepath=None,
                                  plot_size=5,
                                  node_size=200,
                                  cmap=plt.get_cmap('coolwarm'),
                                  ):
    # # Custom colormap.
    # top = plt.get_cmap('Blues_r', 128)
    # bottom = plt.get_cmap('Oranges', 128)
    # newcolors = np.vstack((top(np.linspace(0, 1, 128)),
    #                        bottom(np.linspace(0, 1, 128))))
    # from matplotlib.colors import ListedColormap
    # cmap = ListedColormap(newcolors, name='BlueOrange')

    ftridx = 9  # Select which node feature index to visualize as node colors.

    # Compute all graph perturbations.
    pert_graphs = {}
    for pert_name, pert_func in PERT_FUNC_DICT.items():
        print(f'Computing pert: {pert_name}')

        pert_graph = pert_func(graph.clone())
        if pert_name == 'NodeDegree':
            y = torch.argmax(pert_graph.x, dim=-1)
        elif pert_name == 'NoFeatures':
            y = pert_graph.x[:, 0]
        else:
            y = pert_graph.x[:, ftridx]
        pert_graphs[pert_name] = (pert_graph, y)

    # Compute node color scaling for Bandpass.
    pert_cm = {}
    vals = torch.cat((graph.x[:, ftridx],
                      pert_graphs['BandpassLow'][1],
                      pert_graphs['BandpassMid'][1],
                      pert_graphs['BandpassHigh'][1]),
                     0)
    print(graph.x[:, ftridx])
    print(pert_graphs['BandpassLow'][1])
    print(pert_graphs['BandpassMid'][1])
    print(pert_graphs['BandpassHigh'][1])
    vmin, vmax = vals.min(), vals.max()
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    pert_cm['Bandpass'] = (sm, vmin, vmax)
    print("bandpass:", vmin, vmax)

    # Compute node color scaling for WaveletBank.
    vals = torch.cat((graph.x[:, ftridx],
                      pert_graphs['WaveletBankLow'][1],
                      pert_graphs['WaveletBankMid'][1],
                      pert_graphs['WaveletBankHigh'][1]),
                     0)
    vmin, vmax = vals.min(), vals.max()
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    pert_cm['WaveletBank'] = (sm, vmin, vmax)
    print("wavelet:", vmin, vmax)


    nxg = to_nx(graph, to_undirected=True)
    pos = nx.kamada_kawai_layout(nxg)
    # Plot Original graph.
    y = graph.x[:, ftridx]
    fig = plt.figure(figsize=(plot_size, plot_size))
    ax = fig.gca()
    sm, vmin, vmax = pert_cm['Bandpass']
    print("plot:", vmin, vmax)
    nx.draw(nxg, pos, node_color=y, cmap=cmap, node_size=node_size,
            vmin=vmin, vmax=vmax, ax=ax)
    # cbar = plt.colorbar(sm)
    if savepath is None:
        plt.tight_layout()
        plt.show()
    else:
        checkdir(savepath)
        plt.savefig(os.path.join(savepath, f'Original.pdf'),
                    bbox_inches='tight')
    plt.close(fig)

    # Plot all Perturbations.
    for pert_name, (pert_graph, y) in pert_graphs.items():
        print(f'Plotting pert: {pert_name}')

        nxg_pert = to_nx(pert_graph, to_undirected=True)
        fig = plt.figure(figsize=(plot_size, plot_size))
        ax = fig.gca()

        if pert_name.startswith('NodeDegree'):
            nx.draw(nxg_pert, pos, node_color=y, cmap=plt.get_cmap('viridis'),
                    node_size=node_size, ax=ax)
        elif pert_name.startswith('NoFeatures'):
            nx.draw(nxg_pert, pos, node_color=y, cmap=plt.get_cmap('cubehelix'),
                    node_size=node_size, ax=ax)
        elif 'Frag' in pert_name:
            # Annotate by the cluster assignment.
            idx_components = {u: i for i, node_set in
                              enumerate(nx.connected_components(nxg_pert)) for u in node_set}
            y = [idx_components[u] for u in nxg_pert.nodes()]
            nx.draw(nxg_pert, pos, node_color=y, cmap=plt.get_cmap('tab10'),
                    node_size=node_size, ax=ax)
        elif pert_name.startswith('WaveletBank'):
            sm, vmin, vmax = pert_cm['WaveletBank']
            nx.draw(nxg_pert, pos, node_color=y, cmap=cmap, node_size=node_size,
                    vmin=vmin, vmax=vmax, ax=ax)
            # cbar = plt.colorbar(sm)
        else:
            # Default: use node feature signal as node color
            # but use vmin, vmax value boundaries from bandpass filter
            sm, vmin, vmax = pert_cm['Bandpass']
            nx.draw(nxg_pert, pos, node_color=y, cmap=cmap, node_size=node_size,
                    vmin=vmin, vmax=vmax, ax=ax)
            # cbar = plt.colorbar(sm)

        if savepath is None:
            plt.tight_layout()
            plt.show()
        else:
            checkdir(savepath)
            plt.savefig(os.path.join(savepath, f'{pert_name}.pdf'),
                        bbox_inches='tight')
        plt.close(fig)


def _init():
    cfg.dataset.split_mode = 'random'
    cfg.dataset.task = 'graph'
    cfg.device = 'cpu'
    logging.detail = lambda msg: logging.log(15, msg)


def main():
    _init()
    args = parse_args()

    # Load dataset.
    dataset = load_dataset_master(args.dataset, args.name, '../dataset')

    if 'NodeDegree' in PERT_FUNC_DICT:
        PERT_FUNC_DICT['NodeDegree'] = NodeDegree(dataset,
                                                  is_transductive=False)

    # Generate perturbation comparison plots/
    plot_graph_comparison(
        dataset[args.selected_index],
        savefigfile=args.savefigfile,
    )

    # Plot individual perturbation plots.
    print('W: Individually plotting perts of only the first graph in the list.')
    plot_individual_perturbations(
        dataset[args.selected_index[0]],
        savepath=os.path.dirname(args.savefigfile),
    )

if __name__ == '__main__':
    main()
