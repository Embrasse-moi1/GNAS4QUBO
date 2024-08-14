import networkx as nx
import random
import multiprocessing as mp
import time
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch_geometric.data import DataLoader, Data
from itertools import combinations
import torch_geometric.utils as tgu
from sklearn.model_selection import *
import os.path as osp

def read_label(dir, task):
    labels = None
    nodes = []
    with open(dir + 'edges.txt') as ef:
        for line in ef.readlines():
            nodes.extend(line.strip().split()[:2])
    nodes = sorted(list(set(nodes)))
    node_id_mapping = {old_id: new_id for new_id, old_id in enumerate(nodes)}

    return labels, node_id_mapping


def read_edges(dir, node_id_mapping):
    edges = []
    fin_edges = open(dir + 'edges.txt')
    for line in fin_edges.readlines():
        node1, node2 = line.strip().split()[:2]
        edges.append([node_id_mapping[node1], node_id_mapping[node2]])
    fin_edges.close()
    return edges


def get_degrees(G):
    num_nodes = G.number_of_nodes()
    return np.array([G.degree[i] for i in range(num_nodes)])


def read_file(args, dataname):
    dataset = args.dataset_name
    if dataset in ['brazil-airports', 'europe-airports', 'usa-airports', 'foodweb', 'karate']:
        task = 'node_classification'
    elif dataset in ['arxiv', 'celegans', 'celegans_small', 'facebook', 'ns', 'pb', 'power', 'router', 'usair',
                     'yeast']:
        task = 'LinkPredict'
    elif dataset in ['arxiv_tri', 'celegans_tri', 'celegans_small_tri', 'facebook_tri', 'ns_tri', 'pb_tri', 'power_tri',
                     'router_tri', 'usair_tri', 'yeast_tri']:
        task = 'triplet_prediction'
    elif dataset in ['simulation']:
        task = 'simulation'
    else:
        raise ValueError('dataset not found')
    base_path = osp.dirname(__file__)
    directory = f'{base_path}/dataset/' + dataname + '/'
    labels, node_id_mapping = read_label(directory, task=task)
    edges = read_edges(directory, node_id_mapping)

    G = nx.Graph(edges)

    attributes = np.zeros((G.number_of_nodes(), 1), dtype=np.float32)
    attributes += np.expand_dims(np.log(get_degrees(G) + 1), 1).astype(np.float32)

    G.graph['attributes'] = attributes

    return (G, labels), task


def sample_neg_sets(G, n_samples, set_size):
    neg_sets = []
    n_nodes = G.number_of_nodes()
    max_iter = 1e9
    count = 0
    while len(neg_sets) < n_samples:
        count += 1
        if count > max_iter:
            raise Exception('Reach max sampling number of {}, input graph density too high'.format(max_iter))
        candid_set = [int(random.random() * n_nodes) for _ in range(set_size)]
        for node1, node2 in combinations(candid_set, 2):
            if not G.has_edge(node1, node2):
                neg_sets.append(candid_set)
                break

    return neg_sets


def retain_partial(indices, ratio):
    sample_i = np.random.choice(indices.shape[0], int(ratio * indices.shape[0]), replace=False)
    return indices[sample_i], sample_i


def sample_pos_neg_sets(G, task, data_usage=1.0):
    if task == 'LinkPredict':
        pos_edges = np.array(list(G.edges), dtype=np.int32)
        set_size = 2

    if data_usage < 1 - 1e-6:
        pos_edges, sample_i = retain_partial(pos_edges, ratio=data_usage)
    neg_edges = np.array(sample_neg_sets(G, pos_edges.shape[0], set_size=set_size), dtype=np.int32)
    return pos_edges, neg_edges


def get_mask(idx, length):
    mask = np.zeros(length)
    mask[idx] = 1
    return np.array(mask, dtype=np.int8)


def generate_set_indices_labels(G, task, test_ratio, data_usage=1.0):
    G = G.to_undirected()  # the prediction task completely ignores directions
    pos_edges, neg_edges = sample_pos_neg_sets(G, task,
                                               data_usage=data_usage)  # each shape [n_pos_samples, set_size], note hereafter each "edge" may contain more than 2 nodes
    n_pos_edges = pos_edges.shape[0]
    assert (n_pos_edges == neg_edges.shape[0])
    pos_test_size = int(test_ratio * n_pos_edges)

    set_indices = np.concatenate([pos_edges, neg_edges], axis=0)
    test_pos_indices = random.sample(range(n_pos_edges), pos_test_size)  # randomly pick pos edges for test
    test_neg_indices = list(
        range(n_pos_edges, n_pos_edges + pos_test_size))  # pick first pos_test_size neg edges for test
    test_mask = get_mask(test_pos_indices + test_neg_indices, length=2 * n_pos_edges)
    train_mask = np.ones_like(test_mask) - test_mask
    labels = np.concatenate([np.ones((n_pos_edges,)), np.zeros((n_pos_edges,))]).astype(np.int32)
    G.remove_edges_from(
        [node_pair for set_index in list(set_indices[test_pos_indices]) for node_pair in combinations(set_index, 2)])

    # permute everything for stable training
    permutation = np.random.permutation(2 * n_pos_edges)
    set_indices = set_indices[permutation]
    labels = labels[permutation]
    train_mask, test_mask = train_mask[permutation], test_mask[permutation]

    return G, labels, set_indices, (train_mask, test_mask)


def generate_samples_labels_graph(G, labels, task, args):
    if labels is None:
        G, labels, set_indices, (train_mask, val_test_mask) = generate_set_indices_labels(G, task,
                                                                                          test_ratio=2 * args.test_ratio,
                                                                                          data_usage=args.data_usage)

    return G, labels, set_indices, (train_mask, val_test_mask)


def get_hop_num(prop_depth, layers, max_sprw, feature_flags):
    # TODO: may later use more rw_depth to control as well?
    return int(prop_depth * layers) + 1


def get_features_sp_sample(G, node_set, max_sp):
    dim = max_sp + 2
    set_size = len(node_set)
    sp_length = np.ones((G.number_of_nodes(), set_size), dtype=np.int32) * -1
    for i, node in enumerate(node_set):
        for node_ngh, length in nx.shortest_path_length(G, source=node).items():
            sp_length[node_ngh, i] = length
    sp_length = np.minimum(sp_length, max_sp)
    onehot_encoding = np.eye(dim, dtype=np.float64)  # [n_features, n_features]
    features_sp = onehot_encoding[sp_length].sum(axis=1)
    return features_sp


def get_features_rw_sample(adj, node_set, rw_depth):
    epsilon = 1e-6
    adj = adj / (adj.sum(1, keepdims=True) + epsilon)
    rw_list = [np.identity(adj.shape[0])[node_set]]
    for _ in range(rw_depth):
        rw = np.matmul(rw_list[-1], adj)
        rw_list.append(rw)
    features_rw_tmp = np.stack(rw_list, axis=2)  # shape [set_size, N, F]
    # pooling
    features_rw = features_rw_tmp.sum(axis=0)
    return features_rw


def get_data_sample(G, set_index, hop_num, feature_flags, max_sprw, label, debug=False):
    # first, extract subgraph
    set_index = list(set_index)
    sp_flag, rw_flag = feature_flags
    max_sp, rw_depth = max_sprw
    if len(set_index) > 1:
        G = G.copy()
        G.remove_edges_from(combinations(set_index, 2))
    edge_index = torch.tensor(list(G.edges)).long().t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1, 0],]], dim=-1)
    subgraph_node_old_index, new_edge_index, new_set_index, edge_mask = tgu.k_hop_subgraph(
        torch.tensor(set_index).long(), hop_num, edge_index, num_nodes=G.number_of_nodes(), relabel_nodes=True)

    # reconstruct networkx graph object for the extracted subgraph
    num_nodes = subgraph_node_old_index.size(0)
    new_G = nx.from_edgelist(new_edge_index.t().numpy().astype(dtype=np.int32), create_using=type(G))
    new_G.add_nodes_from(np.arange(num_nodes, dtype=np.int32))  # to add disconnected nodes
    assert (new_G.number_of_nodes() == num_nodes)

    # Construct x from x_list
    x_list = []
    attributes = G.graph['attributes']
    if attributes is not None:
        new_attributes = torch.tensor(attributes, dtype=torch.float32)[subgraph_node_old_index]
        if new_attributes.dim() < 2:
            new_attributes.unsqueeze_(1)
        x_list.append(new_attributes)
    # if deg_flag:
    #     x_list.append(torch.log(tgu.degree(new_edge_index[0], num_nodes=num_nodes, dtype=torch.float32).unsqueeze(1)+1))
    if sp_flag:
        features_sp_sample = get_features_sp_sample(new_G, new_set_index.numpy(), max_sp=max_sp)
        features_sp_sample = torch.from_numpy(features_sp_sample).float()
        x_list.append(features_sp_sample)
    if rw_flag:
        adj = np.asarray(
            nx.adjacency_matrix(new_G, nodelist=np.arange(new_G.number_of_nodes(), dtype=np.int32)).todense().astype(
                np.float32))  # [n_nodes, n_nodes]
        features_rw_sample = get_features_rw_sample(adj, new_set_index.numpy(), rw_depth=rw_depth)
        features_rw_sample = torch.from_numpy(features_rw_sample).float()
        x_list.append(features_rw_sample)

    x = torch.cat(x_list, dim=-1)
    y = torch.tensor([label], dtype=torch.long) if label is not None else torch.tensor([0], dtype=torch.long)
    new_set_index = new_set_index.long().unsqueeze(0)
    if not debug:
        return Data(x=x, edge_index=new_edge_index, y=y, set_indices=new_set_index)
    else:
        return Data(x=x, edge_index=new_edge_index, y=y, set_indices=new_set_index,
                    old_set_indices=torch.tensor(set_index).long().unsqueeze(0),
                    old_subgraph_indices=subgraph_node_old_index)


def parallel_worker(x):
    return get_data_sample(*x)


def extract_subgaphs(G, labels, set_indices, prop_depth, layers, feature_flags, task, max_sprw, parallel, debug=False):
    data_list = []
    hop_num = get_hop_num(prop_depth, layers, max_sprw, feature_flags)
    n_samples = set_indices.shape[0]
    if not parallel:
        for sample_i in tqdm(range(n_samples)):
            data = get_data_sample(G, set_indices[sample_i], hop_num, feature_flags, max_sprw,
                                   label=labels[sample_i] if labels is not None else None, debug=debug)
            data_list.append(data)
    else:
        pool = mp.Pool(4)
        results = pool.map_async(parallel_worker,
                                 [(G, set_indices[sample_i], hop_num, feature_flags, max_sprw,
                                   labels[sample_i] if labels is not None else None, debug) for sample_i in
                                  range(n_samples)])
        remaining = results._number_left
        pbar = tqdm(total=remaining)
        while True:
            pbar.update(remaining - results._number_left)
            if results.ready():
                break
            remaining = results._number_left
            time.sleep(0.2)
        data_list = results.get()
        pool.close()
        pbar.close()
    return data_list


def split_datalist(data_list, masks):
    # generate train_set
    train_mask, val_test_mask = masks
    num_graphs = len(data_list)
    assert ((train_mask.sum() + val_test_mask.sum()).astype(np.int32) == num_graphs)
    assert (train_mask.shape[0] == num_graphs)
    train_indices = np.arange(num_graphs)[train_mask.astype(bool)]
    train_set = [data_list[i] for i in train_indices]
    # generate val_set and test_set
    val_test_indices = np.arange(num_graphs)[val_test_mask.astype(bool)]
    val_test_labels = np.array([data.y for data in data_list], dtype=np.int32)[val_test_indices]
    val_indices, test_indices = train_test_split(val_test_indices, test_size=int(0.5 * len(val_test_indices)),
                                                 stratify=val_test_labels)
    val_set = [data_list[i] for i in val_indices]
    test_set = [data_list[i] for i in test_indices]
    return train_set, val_set, test_set


def load_datasets(train_set, val_set, test_set, bs):
    num_workers = 0
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def get_data(G, task, args, labels):
    G = deepcopy(G)  # to make sure original G is unchanged

    sp_flag = 'sp' in args.feature
    rw_flag = 'rw' in args.feature

    feature_flags = (sp_flag, rw_flag)

    G, labels, set_indices, (train_mask, val_test_mask) = generate_samples_labels_graph(G, labels, task, args)

    data_list = extract_subgaphs(G, labels, set_indices, prop_depth=args.prop_depth, layers=args.layers,
                                 feature_flags=feature_flags, task=task,
                                 max_sprw=(args.max_sp, args.rw_depth), parallel=args.parallel, debug=args.debug)
    train_set, val_set, test_set = split_datalist(data_list, (train_mask, val_test_mask))

    train_loader, val_loader, test_loader = load_datasets(train_set, val_set, test_set, bs=args.bs)

    return (train_loader, val_loader, test_loader), len(np.unique(labels))
