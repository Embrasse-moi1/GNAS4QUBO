import dgl
import numpy as np
import torch
import networkx as nx
import scipy.sparse
from networkx import convert_node_labels_to_integers
device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

def get_edge_index(allRows):

    G = nx.from_edgelist(allRows)
    G = convert_node_labels_to_integers(G, first_label=0, ordering="default", label_attribute=None)

    H = nx.Graph()
    H.add_nodes_from(sorted(G.nodes(data=True)))
    H.add_edges_from(G.edges(data=True))

    graph_dgl = dgl.from_networkx(nx_graph=H)
    graph_dgl = graph_dgl.to(device1)

    n_nodes = len(G.nodes())
    adj_matrix = nx.adjacency_matrix(G)
    coo_matrix = scipy.sparse.coo_matrix(adj_matrix)
    #values = coo_matrix.data  # 取边的权值
    indices = np.vstack((coo_matrix.row, coo_matrix.col))  # 正真需要的coo形式
    edge_index_A = torch.LongTensor(indices)
    edge_index_A = edge_index_A.to(device1)

    return edge_index_A, graph_dgl, G, n_nodes
