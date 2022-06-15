import os
import os.path as osp

import numpy as np
import pandas as pd
import scipy
from matplotlib.colors import ListedColormap
import seaborn as sns
from matplotlib import pyplot as plt, font_manager
from sklearn.decomposition import PCA


def checkdir(directory):
    """Creacte directory if not yet existed, skip if None.
    """
    if directory and not osp.isdir(directory):
        os.makedirs(directory)


def plot_scores(
    score_mat,
    datasets,
    perturbations,
    figsize=(7, 4),
    title=None,
    save_dir=None,
    center=None,
):
    checkdir(save_dir)
    score_df = pd.DataFrame(score_mat)
    score_df.columns = perturbations
    score_df.index = datasets

    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
        filename = fr'{title}.pdf'
    else:
        filename = 'scores.pdf'
    sns.heatmap(score_df, annot=True, fmt='.2f', cbar=False, cmap=sns.diverging_palette(300, 145, as_cmap=True), center=center)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(osp.join(save_dir, filename), bbox_inches='tight')
    plt.show()


def plot_aurocs(
    score_mat,
    datasets,
    perturbations,
    figsize=(14, 6),
    dendrogram_pos_param=(0.285, 2.8),
    score_col_pos=[0.506, 0.072, 0.05, 0.71],
    save_dir=None,
    y_tick_left_shift=0.255,
):
    """Plot dataset taxonomy.

    Args:
        score_mat: full raw score matrix, the first column is the baseline
            performance score (no perturbation applied), and the rest of the
            columns correspond to different perturbations, in the order of
            ``perturbations``, and the rows are in the order of ``datasets``.
        datasets: list of datasets used.
        perturbations: list of perturbations used.
        figsize: size of the final figure.
        dendrogram_pos_param: left-shift, width-scale.
        score_col_pos: left-right, up-down, width, height.
        save_dir: save figure to directory, do not save if None.
        y_tick_left_shift: shifint the y-tick label (datasets) to the left by
            some amount.

    """
    checkdir(save_dir)
    score_diff_percentage_mat = np.nan_to_num(
        score_mat[:, 1:] / score_mat[:, 0:1],
        nan=0,
    )
    score_diff_log2_mat = np.log2(score_diff_percentage_mat)

    # Cluster based on log2 ratio, but annotate with percentage.
    #   Precomputing the clustering outside `sns.clustermap` so that
    #   we can set the method and `optimal_ordering=True`.
    row_linkage = scipy.cluster.hierarchy.linkage(
        score_diff_log2_mat,
        method='ward', metric='euclidean',
        optimal_ordering=True
    )
    fig = sns.clustermap(
        score_diff_log2_mat,
        xticklabels=perturbations[1:],
        yticklabels=[i.ljust(30) for i in datasets],
        cmap=sns.diverging_palette(300, 145, as_cmap=True),
        col_cluster=False,
        row_linkage=row_linkage,
        linewidths=1.5,
        annot=score_diff_percentage_mat,
        fmt='.0%',
        center=0,
        tree_kws={'linewidths': 2},
        figsize=figsize,
    )

    # Get hierarchical clustering order of the datatsets
    taxo_order = fig.dendrogram_row.reordered_ind

    # Extend the figure created by clustermap and plot 1 column heatmap
    fig.fig.subplots_adjust(right=0.5)
    new_ax = fig.fig.add_axes(score_col_pos)  # adjust original score coulumn here
    orig_score_df = pd.DataFrame(
        score_mat[taxo_order, 0] / 100,
        index=[datasets[i] for i in taxo_order]
    )
    sns.heatmap(
        orig_score_df,
        cmap=sns.color_palette('Blues', as_cmap=True),
        linewidths=1.0,
        vmin=0.5,
        vmax=1,
        annot=True, cbar=False,
        xticklabels=['original\nAUROC'],
        ax=new_ax
    )
    new_ax.set_yticklabels([])

    # Format dendrogram of the clustermap
    fig.cax.set_visible(False)
    fig.ax_col_dendrogram.set_visible(False)
    pos1 = fig.ax_row_dendrogram.get_position()  # get the original position
    pos2 = [
        pos1.x0 - dendrogram_pos_param[0],
        pos1.y0,
        pos1.width * dendrogram_pos_param[1],
        pos1.height,
    ]
    fig.ax_row_dendrogram.set_position(pos2)

    # Format axes of both the clustermap and the heatmap
    for i, ax in enumerate(fig.fig.axes):  # getting all axes of the fig object
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha='center',
            fontsize=12
        )
        bbox = dict(ec="grey", fc="lightgrey", alpha=0.29)
        ax.yaxis.tick_left()
        ax.tick_params(axis=u'both', which=u'both', length=0)
        ax.set_yticklabels(
            ax.get_yticklabels(),
            fontsize=12,
            bbox=bbox,
            horizontalalignment='left',
            x=-y_tick_left_shift,
        )

    if save_dir is not None:
        plt.savefig(osp.join(save_dir, 'aurocs.pdf'), bbox_inches='tight')
    plt.show()


def plot_quantized_aurocs(
    score_mat,
    datasets,
    perturbations,
    figsize=(14, 6),
    dendrogram_pos_param=(0.285, 2.8),
    score_col_pos=[0.506, 0.072, 0.05, 0.71],
    save_dir=None,
    y_tick_left_shift=0.255,
):
    """Plot dataset taxonomy.

    Args:
        score_mat: full raw score matrix, the first column is the baseline
            performance score (no perturbation applied), and the rest of the
            columns correspond to different perturbations, in the order of
            ``perturbations``, and the rows are in the order of ``datasets``.
        datasets: list of datasets used.
        perturbations: list of perturbations used.
        figsize: size of the final figure.
        dendrogram_pos_param: left-shift, width-scale.
        score_col_pos: left-right, up-down, width, height.
        save_dir: save figure to directory, do not save if None.
        y_tick_left_shift: shifint the y-tick label (datasets) to the left by
            some amount.

    """
    checkdir(save_dir)
    score_diff_percentage_mat = np.nan_to_num(
        score_mat[:, 1:] / score_mat[:, 0:1],
        nan=0,
    )
    # score_diff_log2_mat = np.log2(score_diff_percentage_mat)
    score_diff_quant_mat = np.copy(score_diff_percentage_mat)
    score_diff_quant_mat = np.where(score_diff_percentage_mat < 0.8, 0, score_diff_quant_mat)
    score_diff_quant_mat = np.where((0.8 <= score_diff_percentage_mat) & (score_diff_percentage_mat < 0.95), 1, score_diff_quant_mat)
    score_diff_quant_mat = np.where((0.95 <= score_diff_percentage_mat) & (score_diff_percentage_mat < 1.05), 2, score_diff_quant_mat)
    score_diff_quant_mat = np.where((1.05 <= score_diff_percentage_mat) & (score_diff_percentage_mat < 1.3), 3, score_diff_quant_mat)
    score_diff_quant_mat = np.where(1.3 <= score_diff_percentage_mat, 4, score_diff_quant_mat)
    score_diff_quant_mat /= 4

    # Cluster based on quantized diff ratios, but annotate with percentage.
    #   Precomputing the clustering outside `sns.clustermap` so that
    #   we can set the method and `optimal_ordering=True`.
    row_linkage = scipy.cluster.hierarchy.linkage(
        score_diff_quant_mat,
        method='ward', metric='euclidean',
        optimal_ordering=True
    )
    fig = sns.clustermap(
        score_diff_quant_mat,
        xticklabels=perturbations[1:],
        yticklabels=[i.ljust(30) for i in datasets],
        cmap=sns.diverging_palette(300, 145, as_cmap=True),
        col_cluster=False,
        row_linkage=row_linkage,
        linewidths=1.5,
        annot=score_diff_percentage_mat,
        fmt='.0%',
        center=0.5,
        tree_kws={'linewidths': 2},
        figsize=figsize,
    )

    # Get hierarchical clustering order of the datatsets
    taxo_order = fig.dendrogram_row.reordered_ind

    # Extend the figure created by clustermap and plot 1 column heatmap
    fig.fig.subplots_adjust(right=0.5)
    new_ax = fig.fig.add_axes(score_col_pos)  # adjust original score coulumn here
    orig_score_df = pd.DataFrame(
        score_mat[taxo_order, 0] / 100,
        index=[datasets[i] for i in taxo_order]
    )
    sns.heatmap(
        orig_score_df,
        cmap=sns.color_palette('Blues', as_cmap=True),
        linewidths=1.0,
        vmin=0.5,
        vmax=1,
        annot=True, cbar=False,
        xticklabels=['original\nAUROC'],
        ax=new_ax
    )
    new_ax.set_yticklabels([])

    # Format dendrogram of the clustermap
    fig.cax.set_visible(False)
    fig.ax_col_dendrogram.set_visible(False)
    pos1 = fig.ax_row_dendrogram.get_position()  # get the original position
    pos2 = [
        pos1.x0 - dendrogram_pos_param[0],
        pos1.y0,
        pos1.width * dendrogram_pos_param[1],
        pos1.height,
    ]
    fig.ax_row_dendrogram.set_position(pos2)

    # Format axes of both the clustermap and the heatmap
    for i, ax in enumerate(fig.fig.axes):  # getting all axes of the fig object
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha='center',
            fontsize=12
        )
        bbox = dict(ec="grey", fc="lightgrey", alpha=0.29)
        ax.yaxis.tick_left()
        ax.tick_params(axis=u'both', which=u'both', length=0)
        ax.set_yticklabels(
            ax.get_yticklabels(),
            fontsize=12,
            bbox=bbox,
            horizontalalignment='left',
            x=-y_tick_left_shift,
        )

    if save_dir is not None:
        plt.savefig(osp.join(save_dir, 'quant_aurocs.pdf'), bbox_inches='tight')
    plt.show()


def plot_pca(
    score_mat,
    datasets,
    perturbations,
    figsize=(12, 12),
    pca_ylimit=(None, None),
    pca_xlimit=(None, None),
    plot_components=True,
    plot_variance=True,
    plot_stdz=True,
    save_dir=None,
    **kwargs
):
    checkdir(save_dir)
    score_diff_mat = np.nan_to_num(score_mat[:, 1:] / score_mat[:, 0:1], nan=0)
    score_diff_mat = np.log2(score_diff_mat)

    pca = PCA(n_components=min([len(perturbations), len(datasets)]) - 1)
    pca.fit(score_diff_mat)
    X = pca.transform(score_diff_mat)[:, :2]

    plt.figure(figsize=figsize)
    # sns.set_context("paper")
    sns.scatterplot(*X.T, **kwargs)
    sns.despine()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.ylim(*pca_ylimit)
    plt.xlim(*pca_xlimit)

    # Adjust the legend
    font = font_manager.FontProperties(family='monospace',
                                       style='normal', size=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    # Sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    lgnd = plt.gca().legend(handles, labels, prop=font)
    for handle in lgnd.legendHandles:
        handle.set_sizes([100])

    for i, txt in enumerate(datasets):
        plt.annotate(txt, (X[i, 0] + 0.02, X[i, 1] - 0.005))  # Node-level tasks
        # plt.annotate(txt, (X[i, 0] + 0.03, X[i, 1] - 0.02))  # Graph-level tasks

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(osp.join(save_dir, 'pca.pdf'), bbox_inches='tight')
    plt.show()

    # Visualize components
    pc_lables = [f'PC{i+1} ({ev*100:.0f}%)'
                 for i, ev in enumerate(pca.explained_variance_ratio_)]
    if plot_components:
        plt.figure(figsize=(7, 2))
        sns.heatmap(pca.components_[:2],
                    xticklabels=perturbations[1:],
                    yticklabels=pc_lables[:2],
                    center=0,
                    # cmap=sns.diverging_palette(220, 20, as_cmap=True),
                    cmap=sns.color_palette("vlag", as_cmap=True),
                    # vmin=-1, vmax=1,
                    )
        # adjust x-axis labels
        ax = plt.gca()
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha='center',
            fontsize=12
        )
        ax.tick_params(axis=u'both', which=u'both', length=0)
        ax.set_yticklabels(
            ax.get_yticklabels(),
            fontsize=12,
            rotation=0,
        )
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(osp.join(save_dir, 'pca_comps.pdf'), bbox_inches='tight')
        plt.show()

    # Explained variance
    if plot_variance:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))

        axes[0].plot(pca.explained_variance_ratio_)
        axes[0].set_title("Explained variance")

        axes[1].plot(np.cumsum(pca.explained_variance_ratio_))
        axes[1].set_title("Total explained variance")

        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(osp.join(save_dir, 'vars.pdf'), bbox_inches='tight')
        plt.show()

        print(np.cumsum(pca.explained_variance_ratio_))

    if plot_stdz:
        plt.figure(figsize=(5, 5))
        score_diff_mat_stdz = (
                (score_diff_mat - score_diff_mat.mean(axis=0))
                / score_diff_mat.std(axis=0)
        )
        sns.heatmap(
            score_diff_mat_stdz.T @ score_diff_mat_stdz / len(datasets),
            annot=True,
            fmt='.1f',
            xticklabels=perturbations[1:],
            yticklabels=perturbations[1:],
            cbar=False
        )
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(osp.join(save_dir, 'stdz.pdf'), bbox_inches='tight')
        plt.show()


def plot_comparisons(df, names, title, save_dir=None, figsize=None):
    perturbations = [
        '-',
        'NoFeat',
        'NDeg',
        'NoEdges',
        'FConn',
        'Frag-k1',
        'Frag-k2',
        'Frag-k3',
        'CSpars',
        'Fiedler',
    ]

    index = [
        'CIFAR10',
        'CLUSTER',
        'ENZYMES',
        'IMDB-INARY',
        'PROTEINS'
    ]

    # mapping from letters to numbers
    letter2num = dict(zip(names, np.arange(len(names))))
    letter2num['0'] = 9 # arbitrary value
    df2 = pd.DataFrame(np.array([letter2num[i] for i in df.values.flat]).reshape(df.shape))

    if figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(df2.values, vmin=0, cmap='tab20')

    for i in range(len(df2)):
        for j in range(len(df2.columns)):
            ax.text(j, i, df.values[i, j], ha="center", va="center")

    ax.set_xticks(range(df.columns.size))  # <--- set the ticks first
    ax.set_xticklabels(perturbations)

    ax.set_yticks(range(df.index.size))  # <--- set the ticks first
    ax.set_yticklabels(index)

    ax.set_title(title)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, fr'{title}_BEST.png'), bbox_inches='tight')
    plt.show()
