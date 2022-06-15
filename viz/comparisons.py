import numpy as np
import pandas as pd


name_dict = {
    'PyG-TUDataset_DD': 'D&D',
    'PyG-TUDataset_ENZYMES': 'ENZYMES',
    'PyG-TUDataset_PROTEINS': 'PROTEINS',
    'PyG-TUDataset_NCI1': 'NCI1',
    'PyG-TUDataset_IMDB-BINARY': 'IMDB-BINARY',
    'PyG-GNNBenchmarkDataset_MNIST': 'MNIST',
    'PyG-GNNBenchmarkDataset_CIFAR10': 'CIFAR10',
    'PyG-GNNBenchmarkDataset_PATTERN': 'PATTERN',
    'PyG-GNNBenchmarkDataset_CLUSTER': 'CLUSTER',
    }

pert_dict = {
    'none': '-',
    'Fragmented-k1': 'Frag-k1',
    'Fragmented-k2': 'Frag-k2',
    'Fragmented-k3': 'Frag-k3',
    'FullyConnected': 'FConn',
    'NoEdges': 'NoEdges',
    'NoFeatures': 'NoFeat',
    'NodeDegree': 'NodeDeg',
    'ClusterSparsification': 'CSpars',
    'FiedlerFragmentation': 'Fiedler',
}

perturbations = [
    '-',
    'NoFeat',
    'NodeDeg',
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

score_type = 'auc'
score_name = f'score-{score_type}'


def read_df(results_file):
    df = pd.read_json(results_file)
    df['Dataset'] = df['Dataset'].apply(name_dict.get)
    df['Perturbation'] = df['Perturbation'].apply(pert_dict.get)
    df = df[df['Dataset'].notna()]
    return df


def set_mat(df):
    datasets = sorted(df['Dataset'].unique())
    score_mat = np.zeros((len(datasets), len(perturbations)))
    score_mat[:] = np.nan

    for dataset, group in df.groupby('Dataset'):
        dataset_idx = datasets.index(dataset)

        for perturbation, score in group[['Perturbation', score_name]].values:
            perturbation_idx = perturbations.index(perturbation)
            score_mat[dataset_idx, perturbation_idx] = score * 100
    return score_mat


def compare_dfs(rf1, rf2):
    scores_1 = set_mat(read_df(rf1))
    scores_2 = set_mat(read_df(rf2))
    diff = scores_1 - scores_2
    diff_norm = diff - np.mean(diff)
    return diff_norm


def best_dfs(rf_dict):

    scores_dict = {}
    for k in rf_dict.keys():
        df_temp = read_df(rf_dict[k])
        scores_dict[k] = set_mat(df_temp)
    arr_shape = scores_dict[k].shape
    best_dataset = []
    for i in range(arr_shape[0]):
        best_pert = []
        for j in range(arr_shape[1]):
            best = 0
            best_k = '0'
            for k, v in scores_dict.items():
                if v[i][j] > best:
                    best = v[i][j]
                    best_k = k
            best_pert.append(best_k)
        best_dataset.append(best_pert)

    df_best = pd.DataFrame(best_dataset, columns=perturbations, index=index)
    return df_best