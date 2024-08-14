#from llm4gnas.trainer import TrainerBase
from torch import Tensor
from llm4gnas.search_space import GNNBase, Nas_bench_graph_co_GNN
import csv
from typing import Union
import numpy as np
import torch
import dgl
from itertools import chain
import networkx as nx
import scipy.sparse
from networkx.relabel import convert_node_labels_to_integers
from torch_geometric.data import Data
from pyqubo import Array
import torch.nn as nn
from llm4gnas.register import model_factory
from easydict import EasyDict as edict

def create_mis_model(graph, penalty=2):
    N = graph.number_of_nodes()
    X = Array.create("X", shape=(N,), vartype="BINARY")

    hamiltonian = -sum(X)
    for u, v in graph.edges:
        hamiltonian += penalty * (X[u] * X[v])

    return hamiltonian.compile()


###Add by wujunxian
def create_max_cut_model(graph):
    N = graph.number_of_nodes()  # 计算图中节点数目
    X = Array.create("X", shape=(N,), vartype="BINARY")  # 将图中每个节点关联一个二元决策变量X

    hamiltonian = 0
    for u, v in graph.edges:  # 对边进行筛选
        hamiltonian -= (X[u] - X[v]) ** 2

    return hamiltonian.compile()


def create_Q_matrix(graph, is_max_cut=True):
    if is_max_cut:
        model = create_max_cut_model(graph)  # hamiltonian
    else:
        model = create_mis_model(graph)
    N = graph.number_of_nodes()
    extract_val = lambda x: int(x[2:-1])  # 从变量中提取整数部分
    Q_matrix = np.zeros((N, N))
    qubo_dict, _ = model.to_qubo()  # 利用哈密顿量与目标图的信息，计算QUBO的问题矩阵Q
    for (a, b), quv in qubo_dict.items():  # (a, b) = ('X[1]', 'X[0]'), quv = 2.0
        u = min(extract_val(a), extract_val(b))
        v = max(extract_val(a), extract_val(b))
        Q_matrix[u, v] = quv  # 创造问题矩阵Q
    Q_matrix = torch.tensor(Q_matrix, dtype=torch.float32)

    return Q_matrix

class TrainerBase(object):
    def __init__(self, config: dict, **kwargs):
        self.config = config
        self.gnn = None
        self.optimizer = None
        self.device = config.device

    def fit(self, data: Data, gnn: GNNBase, config: dict = None) -> GNNBase:
        pass

    def evaluate(self, data: Data, gnn: Union[GNNBase, None] = None) -> dict:
        # return metrics
        raise NotImplementedError

    def predict(self, data: Data, gnn: Union[GNNBase, None] = None) -> Tensor:
        raise NotImplementedError

class COTrainer(TrainerBase):
    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config

    def evaluate(self, data: Data, model: Union[GNNBase, None] = None) -> dict:
        reader = csv.reader(data)
        allRows = [list(map(int, row[0].split())) for row in reader]  # allRows is a list of Graph
        allRows = allRows[1:]
        for i in range(len(allRows)):
            del allRows[i][2]
        allRows = list(map(tuple, allRows))
        # print(allRows)
        G = nx.from_edgelist(allRows)
        G = convert_node_labels_to_integers(G, first_label=0, ordering="default", label_attribute=None)
        H = nx.Graph()
        H.add_nodes_from(sorted(G.nodes(data=True)))
        H.add_edges_from(G.edges(data=True))
        graph_dgl = dgl.from_networkx(nx_graph=H)
        graph_dgl = graph_dgl
        n_nodes = len(G.nodes())
        A = np.array(nx.to_numpy_array(G))
        Q = create_Q_matrix(G)

        # 将图数据神经网络的输入
        adj_matrix = nx.adjacency_matrix(G)
        coo_matrix = scipy.sparse.coo_matrix(adj_matrix)
        indices = np.vstack((coo_matrix.row, coo_matrix.col))  # 正真需要的coo形式
        edge_index_A = torch.LongTensor(indices)
        edge_index_A = edge_index_A
        dim_embedding = self.config.in_dim
        embed = nn.Embedding(n_nodes, dim_embedding)
        # embed = embed.type(dtype).to(device1)
        inputs = embed.weight
        data = Data(x=inputs, edge_index=edge_index_A)

        opt_params = {'lr': self.config.lr}
        # print(opt_params)

        params = chain(model.parameters(), embed.parameters())
        optimizer = torch.optim.Adam(params, **opt_params)
        IterNUM = 5
        cut_vals = []

        for i in range(IterNUM):
            print(i)
            print(f'Runing model:{model.desc}')

            prev_loss = 1.
            count = 0

            losses = []
            epochs = []

            best_bitstring = torch.zeros((graph_dgl.number_of_nodes(),)).type(Q.dtype).to(
                Q.device)  # 初始化全为0一个二进制张量，将图中每个节点关联一个二进制变量x
            best_loss = model.loss(prob=best_bitstring.float(), Q=Q)

            for epoch in range(self.config.number_epochs):
                probs = model(data)[:, 0]
                loss = model.loss(prob=probs, Q=Q)
                loss_ = loss.detach().item()

                bitstring = (probs.detach() >= self.config.prob_threshold) * 1
                if loss < best_loss:
                    best_loss = loss
                    best_bitstring = bitstring

                if epoch % self.config.out == 0:
                    print(f'Epoch:{epoch}, loss:{loss_}')
                    losses.append(loss_)
                    epochs.append(epoch)

                if (abs(loss_ - prev_loss) <= self.config.tol) | ((loss_ - prev_loss) > 0):
                    count += 1
                else:
                    count = 0

                if count >= self.config.patience:
                    print(f'Stopping early on epoch {epoch}(patience: {self.config.patience})')

                prev_loss = loss_

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bitstring_list = list(best_bitstring)
            # 将张量转换为Long类型
            best_bitstring_long = best_bitstring.long()
            Q_long = Q.long()

            # 对张量进行转置
            transposed_bitstring = best_bitstring_long.permute(*torch.arange(best_bitstring_long.ndim - 1, -1, -1))

            # 执行矩阵乘法
            cut_value_from_training = -(transposed_bitstring @ Q_long @ best_bitstring_long)

            # cut_value_from_training = -(best_bitstring.long().permute(*torch.arange(best_bitstring.ndim - 1, -1, -1)) @ Q @ best_bitstring.long())

            # cut_value_from_training = -(best_bitstring.T @ Q @ best_bitstring)
            cut_vals.append(cut_value_from_training)
        result = max(cut_vals)
        metric = model.metric(maxcut=result, total_edges=4694)
        print(f'The metric of :{model.desc} is :{metric}')

        return metric

model_factory["co_trainer"] = COTrainer