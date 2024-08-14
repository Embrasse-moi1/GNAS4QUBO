import numpy as np
from pyqubo import Array
import torch
import networkx as nx
import jax.numpy as jnp

def create_max_cut_model(graph):
    N = graph.number_of_nodes()     #计算图中节点数目
    X = Array.create("X", shape=(N,), vartype="BINARY")     #将图中每个节点关联一个二元决策变量X

    hamiltonian = 0
    for u, v in graph.edges:    #对边进行筛选
        hamiltonian -= (X[u] - X[v]) ** 2

    return hamiltonian.compile()

def create_mis_model(graph, penalty=2):
    N = graph.number_of_nodes()
    X = Array.create("X", shape=(N,), vartype="BINARY")

    hamiltonian = -sum(X)
    for u, v in graph.edges:
        hamiltonian += penalty * (X[u] * X[v])

    return hamiltonian.compile()

def create_Q_matrix(graph, is_max_cut=True):
    if is_max_cut:
        model = create_max_cut_model(graph)     #hamiltonian
    else:
        model = create_mis_model(graph)
    N = graph.number_of_nodes()
    extract_val = lambda x: int(x[2:-1])    #从变量中提取整数部分
    Q_matrix = np.zeros((N, N))
    qubo_dict, _ = model.to_qubo()      #利用哈密顿量与目标图的信息，计算QUBO的问题矩阵Q
    for (a, b), quv in qubo_dict.items():   #(a, b) = ('X[1]', 'X[0]'), quv = 2.0
        u = min(extract_val(a), extract_val(b))
        v = max(extract_val(a), extract_val(b))
        Q_matrix[u, v] = quv    #创造问题矩阵Q
    Q_matrix = torch.tensor(Q_matrix, dtype=torch.float32)

    return Q_matrix

def loss_func(probs, Q_matrix):
    probs_ = torch.unsqueeze(probs, 1)  #将probs的维度扩展至（N，1）,才能与矩阵Q_matric相乘
    cost = (probs_.T @ Q_matrix @ probs_).squeeze()  #@表示矩阵乘法
    return cost

