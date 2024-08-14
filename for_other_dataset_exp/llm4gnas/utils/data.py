import random
from typing import Dict, Tuple
from for_other_dataset_exp.llm4gnas.utils.get_lp_data import *
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.data import Data, Dataset
import os
import os.path as osp
import pandas as pd
import torch_geometric.transforms as T
import torch
from torch_geometric.data import InMemoryDataset
import yaml
from easydict import EasyDict as edict
import numpy as np
from torch_geometric.loader import DataLoader
from for_other_dataset_exp.llm4gnas.utils import data_util

def check_files_exist(data_path, task_name):
    required_files = ['edge_index.csv', 'graph_info.yaml']
    for file in required_files:
        if not os.path.exists(os.path.join(data_path, "raw", file)):
            raise FileNotFoundError(f"Error: necessary file '{file}' not exist")
    if task_name == "NodeClassification":
        if not os.path.exists(os.path.join(data_path, "raw", 'node_label.csv')):
            raise FileNotFoundError("Error: necessary files node-label.csv not exist")
    elif task_name == "GraphClassification":
        if not os.path.exists(os.path.join(data_path, "raw", 'graph_label.csv')):
            raise FileNotFoundError("Error: necessary files graph-label.csv not exist")
    return True


def load_data_file(data_path, task_name):
    check_files_exist(data_path, task_name)
    # deal file_path
    node_feat_file = os.path.join(data_path, "raw", 'node_feat.csv')
    edge_index_file = os.path.join(data_path, "raw", 'edge_index.csv')
    node_label_file = os.path.join(data_path, "raw", 'node_label.csv')
    graph_label_file = os.path.join(data_path, "raw", 'graph_label.csv')
    num_node_edge_list_file = os.path.join(data_path, "raw", 'num_node_edge_list.csv')
    graph_info_file = os.path.join(data_path, "raw", 'graph_info.yaml')

    try:
        if task_name in ["NodeClassification", "LinkPredict", "GraphClassification"]:
            with open(graph_info_file, 'r') as file:
                graph_info = yaml.safe_load(file)

            # Create a mapping from index to node_id
            node_data = pd.read_csv(node_feat_file)
            node_mapping = {node_id: index for index, node_id in enumerate(node_data['node_id'].unique())}

            # Map the src_index and des_index using the node_id mapping
            edge_data = pd.read_csv(edge_index_file)
            edge_data['src_id'] = edge_data['src_id'].map(node_mapping)
            edge_data['des_id'] = edge_data['des_id'].map(node_mapping)

            if task_name == "NodeClassification":
                # Load node_label data from 'node_label.csv'
                node_label_data = pd.read_csv(node_label_file)
                # Create a mapping from node_id to node_label
                node_label_mapping = {row['node_id']: row['node_label'] for _, row in node_label_data.iterrows()}
                # mapping index to node_label
                node_data['node_label'] = node_data['node_id'].map(node_label_mapping)
                return node_data, edge_data, graph_info

            if task_name == "GraphClassification":
                # Load each graph node && edge number from 'num_node_edge_list.csv'
                graph_label_data = pd.read_csv(graph_label_file)
                graph_label_mapping = {row['graph_id']: row['graph_label'] for _, row in graph_label_data.iterrows()}
                graph_label = graph_label_data['graph_id'].map(graph_label_mapping)

                # Load node_label data from 'graph_label.csv'
                num_node_edge_list = pd.read_csv(num_node_edge_list_file)
                num_node_mapping = {row['graph_id']: row['num_node'] for _, row in num_node_edge_list.iterrows()}
                num_edge_mapping = {row['graph_id']: row['num_edge'] for _, row in num_node_edge_list.iterrows()}
                num_node_edge_list['num_node'] = num_node_edge_list['graph_id'].map(num_node_mapping)
                num_node_edge_list['num_edge'] = num_node_edge_list['graph_id'].map(num_edge_mapping)

                return node_data, edge_data, num_node_edge_list, graph_label, graph_info

        else:
            raise RuntimeError(f"task {task_name} is not supported")

    except RuntimeError:
        print("load data file fail, please check your data_file")


def load_local_data(data_path, task_name):
    check_files_exist(data_path, task_name)
    graph_list = []
    try:
        if task_name in ["NodeClassification", "LinkPredict"]:
            graph = dict()

            # load_data_file
            node_data, edge_data, graph_info = load_data_file(data_path, task_name)

            # deal node_feat
            node_feat_list = [
                [float(x.strip()) for x in node_feat_str.strip('[]').split(',')]
                for node_feat_str in node_data['node_feat']
            ]
            graph['node_feat'] = torch.tensor(node_feat_list, dtype=torch.float32)

            # deal edge_index
            src_tensor = torch.tensor(edge_data['src_id'].values, dtype=torch.long)
            des_tensor = torch.tensor(edge_data['des_id'].values, dtype=torch.long)
            graph['edge_index'] = torch.stack((src_tensor, des_tensor), dim=0)
            data = Data(x=graph['node_feat'], edge_index=graph['edge_index'])

            # deal edge_feat
            if 'edge_feat' in edge_data.columns:
                edge_feat_list = [
                    [float(x.strip()) for x in edge_feat_str.strip('[]').split(',')]
                    for edge_feat_str in edge_data['edge_feat']
                ]
                graph['edge_feat'] = torch.tensor(edge_feat_list, dtype=torch.float32)
                data.edge_attr = graph['edge_feat']

            # node_label
            if task_name == "NodeClassification":
                node_label = node_data['node_label'].values
                graph['node_label'] = torch.tensor(node_label, dtype=torch.long)
                data.y = graph['node_label']

            graph_list.append(data)

        elif task_name in ["GraphClassification"]:
            # load_data_file
            node_data, edge_data, num_node_edge_list, graph_label, graph_info = load_data_file(data_path, task_name)
            graph = dict()
            # deal node_feat
            node_feat_list = [
                [float(x.strip()) for x in node_feat_str.strip('[]').split(',')]
                for node_feat_str in node_data['node_feat']
            ]
            graph['node_feat'] = torch.tensor(node_feat_list, dtype=torch.float32)

            # deal edge_index
            src_tensor = torch.tensor(edge_data['src_id'].values, dtype=torch.long)
            des_tensor = torch.tensor(edge_data['des_id'].values, dtype=torch.long)
            graph['edge_index'] = torch.stack((src_tensor, des_tensor), dim=0)

            # deal graph_label
            graph_label = graph_label['graph_label'].values
            graph['graph_label'] = torch.tensor(graph_label, dtype=torch.long)

            # deal edge_feat
            if 'edge_feat' in edge_data.columns:
                edge_feat_list = [
                    [float(x.strip()) for x in edge_feat_str.strip('[]').split(',')]
                    for edge_feat_str in edge_data['edge_feat']
                ]
                graph['edge_feat'] = torch.tensor(edge_feat_list, dtype=torch.float32)

            num_edge = 0
            num_node = 0
            # split to graph
            for i in range(len(num_node_edge_list['graph_id'])):
                current_num_node = num_node_edge_list['num_node'][i]
                current_num_edge = num_node_edge_list['num_edge'][i]
                data = Data(x=graph['node_feat'][num_node:num_node + current_num_node],
                            edge_index=graph['edge_index'][num_edge:num_edge + current_num_edge],
                            y=graph['graph_label'][i])
                num_node += current_num_node
                num_edge += current_num_edge
                if 'edge_feat' in edge_data.columns:
                    data.edge_attr = graph['edge_feat'][num_edge:num_edge + current_num_edge]

                graph_list.append(data)

        else:
            raise RuntimeError(f"task {task_name} is not supported")

    except RuntimeError:
        print("save graph list fail")

    return graph_list


node_transform = T.Compose([T.ToUndirected(), T.RandomNodeSplit(
    split="train_rest",
    num_val=0.2,
    num_test=0.2,
)])


class NCdataset(InMemoryDataset):
    def __init__(self, data_path, task_name="NodeClassification", root='./dataset', transform=node_transform,
                 pre_transform=None, pre_filter=None):
        self.data_path = data_path
        self.task_name = task_name
        self.name = os.path.basename(os.path.normpath(self.data_path))
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        files_and_folders = os.listdir(self.raw_dir())
        files = [f for f in files_and_folders if os.path.isfile(os.path.join(self.raw_dir(), f))]
        names = []
        for file in files:
            names.append(file)
        return [f'ind.{self.name.lower()}.{name}' for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        # Read data into huge `Data` list.
        data_list = load_local_data(self.data_path, self.task_name)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


class LPdataset(InMemoryDataset):
    def __init__(self, data_path, task_name="LinkPrediction", root='../dataset', transform=None, pre_transform=None,
                 pre_filter=None):
        self.data_path = data_path
        self.task_name = task_name
        self.name = os.path.basename(os.path.normpath(self.data_path))
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        files_and_folders = os.listdir(self.raw_dir())
        files = [f for f in files_and_folders if os.path.isfile(os.path.join(self.raw_dir(), f))]
        names = []
        for file in files:
            names.append(file)
        return [f'ind.{self.name.lower()}.{name}' for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        # Read data into huge `Data` list.
        data_list = load_local_data(self.data_path, self.task_name)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


class GCdataset(InMemoryDataset):
    def __init__(self, data_path, task_name="GraphClassification", root='../dataset', transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.data_path = data_path
        self.task_name = task_name
        self.name = os.path.basename(os.path.normpath(self.data_path))
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        files_and_folders = os.listdir(self.raw_dir())
        files = [f for f in files_and_folders if os.path.isfile(os.path.join(self.raw_dir(), f))]
        names = []
        for file in files:
            names.append(file)
        return [f'ind.{self.name.lower()}.{name}' for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        # Read data into huge `Data` list.
        data_list = load_local_data(self.data_path, self.task_name)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


def load_data(args):
    base_path = osp.dirname(__file__)
    if args.graph_type == "homo":
        if args.input != "":
            if args.task_name == "NodeClassification":
                dataset = NCdataset(args.input, args.task_name)
            elif args.task_name == "LinkPredict":
                dataset = LPdataset(args.input, args.task_name)
            elif args.task_name == "GraphClassification":
                dataset = GCdataset(args.input, args.task_name)
            else:
                raise RuntimeError(f"this {args.task_name} has no been suppored in custom dataset")
        elif args.dataset_name.lower() in ["cora", "citeseer", "pubmed"]:
            dataset = Planetoid(f"{base_path}/../../dataset/", args.dataset_name.lower()).to(args.device)
        elif args.dataset_name.upper() in ["MUTAG", "IMDB-BINARY", "IMDB-MULTI", "PROTEINS"]:
            dataset = TUDataset(f"{base_path}/../../dataset/", args.dataset_name.upper()).to(args.device)
        elif args.task_name == 'LinkPredict':
            (G, labels), task = read_file(args, args.dataset_name)
            dataloaders, out_features = get_data(G, task=task, labels=labels, args=args)
            #train_loader, val_loader, test_loader = dataloaders
            dataset = dataloaders
        else:
            raise RuntimeError("no dataset")
        args.in_dim = max(dataset.num_node_features, 1) if args.task_name != 'LinkPredict' else 53
        args.out_dim = dataset.num_classes if args.task_name != 'LinkPredict' else 2

    elif args.graph_type == "hetero":
        dataset = data_util.load_data(args.dataset_name, args.task_name)

    return dataset


def get_optimizer(model, args: Dict) -> torch.optim.Optimizer:
    optim = args.optimizer
    if optim == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif optim == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError


def get_loader(dataset: Dataset, args: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if args.task_name == 'GraphClassification':
        dataset = dataset.shuffle()
        data_size = len(dataset)
        train_dataset = dataset[:int(data_size * args.train_ratio)]
        val_dataset = dataset[int(data_size * args.train_ratio):int(data_size * (args.train_ratio + args.val_ratio))]
        test_dataset = dataset[int(data_size * (args.train_ratio + args.val_ratio)):]
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        train_loader = val_loader = test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def partition_dataset(data):
    data.num_nodes = len(data.y)
    num_nodes = data.num_nodes
    node_id = np.arange(num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(num_nodes * 0.6):int(num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(num_nodes * 0.8):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(num_nodes)])
    return data


if __name__ == '__main__':
    base_path = osp.dirname(__file__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = edict({
        "input": f"{base_path}/dataset/arxiv2023",
        "task_name": "NodeClassification",
    })
    dataset = load_data(config)
    dataset = partition_dataset(dataset)
    print(dataset)
    print(dataset[0])
    print(dataset.train_mask)
