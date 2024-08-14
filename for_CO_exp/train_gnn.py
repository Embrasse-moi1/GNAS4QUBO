from time import time
import csv
from load_data import get_edge_index
import jax.numpy as jnp
import networkx as nx
from co_corefunc import create_Q_matrix
import create_gnn
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, ChebConv, ARMAConv, GraphConv
import torch.nn.functional as F
device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

class MyGraphNetwork0000(nn.Module):
    option_list = None
    def __init__(self, in_features, out_features,hidden_dim=5, cheb_order=2, dropout=0.1):
        super(MyGraphNetwork0000, self).__init__()
        self.gcn = GCNConv(in_features, hidden_dim).to(device1)
        self.gat = GATConv(in_features, hidden_dim).to(device1)
        self.sage = SAGEConv(in_features, hidden_dim).to(device1)
        self.gin = GINConv(nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU())).to(device1)
        self.cheb = ChebConv(in_features, hidden_dim, cheb_order).to(device1)
        self.arma = ARMAConv(in_features, hidden_dim).to(device1)
        self.graph = GraphConv(in_features, hidden_dim).to(device1)
        self.skip = nn.Linear(in_features, hidden_dim).to(device1)
        self.fc = nn.Linear(in_features, hidden_dim).to(device1)
        self.fc_out = nn.Linear(hidden_dim * 4, out_features).to(device1)
        self.dropout_frac = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        case = {
            'gcn': self.gcn(x, edge_index),
            'gat': self.gat(x, edge_index),
            'sage': self.sage(x, edge_index),
            'gin': self.gin(x, edge_index),
            'cheb': self.cheb(x, edge_index),
            'arma': self.arma(x, edge_index),
            'graph': self.graph(x, edge_index),
            'skip': self.skip(x),
            'fc': self.fc(x),
        }

        x_1 = case.get(self.option_list[0])
        x_1 = torch.relu(x_1)
        x_1 = F.dropout(x_1, p=self.dropout_frac)

        x_2 = case.get(self.option_list[1])
        x_2 = torch.relu(x_2)
        x_2 = F.dropout(x_2, p=self.dropout_frac)

        x_3 = case.get(self.option_list[2])
        x_3 = torch.relu(x_3)
        x_3 = F.dropout(x_3, p=self.dropout_frac)

        x_4 = case.get(self.option_list[3])
        x_4 = torch.relu(x_4)
        x_4 = F.dropout(x_4, p=self.dropout_frac)

        output = torch.cat((x_1, x_2, x_3, x_4), dim=1)
        output = self.fc_out(output)
        output = torch.sigmoid(output)

        return output

class MyGraphNetwork0001(nn.Module):
    option_list = None
    def __init__(self, in_features, out_features,hidden_dim=5, cheb_order=2, dropout=0.1):
        super(MyGraphNetwork0001, self).__init__()
        self.gcn = GCNConv(in_features, hidden_dim).to(device1)
        self.gat = GATConv(in_features, hidden_dim).to(device1)
        self.sage = SAGEConv(in_features, hidden_dim).to(device1)
        self.gin = GINConv(nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU())).to(device1)
        self.cheb = ChebConv(in_features, hidden_dim, cheb_order).to(device1)
        self.arma = ARMAConv(in_features, hidden_dim).to(device1)
        self.graph = GraphConv(in_features, hidden_dim).to(device1)
        self.skip = nn.Linear(in_features, hidden_dim).to(device1)
        self.fc = nn.Linear(in_features, hidden_dim).to(device1)

        self.gcn1 = GCNConv(hidden_dim, hidden_dim).to(device1)
        self.gat1 = GATConv(hidden_dim, hidden_dim).to(device1)
        self.sage1 = SAGEConv(hidden_dim, hidden_dim).to(device1)
        self.gin1 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())).to(device1)
        self.cheb1 = ChebConv(hidden_dim, hidden_dim, cheb_order).to(device1)
        self.arma1 = ARMAConv(hidden_dim, hidden_dim).to(device1)
        self.graph1 = GraphConv(hidden_dim, hidden_dim).to(device1)
        self.skip1 = nn.Linear(hidden_dim, hidden_dim).to(device1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim).to(device1)
        self.fc_out = nn.Linear(hidden_dim * 3, out_features).to(device1)
        self.dropout_frac = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        case = {
            'gcn': self.gcn(x, edge_index),
            'gat': self.gat(x, edge_index),
            'sage': self.sage(x, edge_index),
            'gin': self.gin(x, edge_index),
            'cheb': self.cheb(x, edge_index),
            'arma': self.arma(x, edge_index),
            'graph': self.graph(x, edge_index),
            'skip': self.skip(x),
            'fc': self.fc(x),
        }

        x_1 = case.get(self.option_list[0])
        x_1 = torch.relu(x_1)
        x_1 = F.dropout(x_1, p=self.dropout_frac)

        x_2 = case.get(self.option_list[1])
        x_2 = torch.relu(x_2)
        x_2 = F.dropout(x_2, p=self.dropout_frac)

        x_3 = case.get(self.option_list[2])
        x_3 = torch.relu(x_3)
        x_3 = F.dropout(x_3, p=self.dropout_frac)

        case2 = {
            'gcn': self.gcn1(x_1, edge_index),
            'gat': self.gat1(x_1, edge_index),
            'sage': self.sage1(x_1, edge_index),
            'gin': self.gin1(x_1, edge_index),
            'cheb': self.cheb1(x_1, edge_index),
            'arma': self.arma1(x_1, edge_index),
            'graph': self.graph1(x_1, edge_index),
            'skip': self.skip1(x_1),
            'fc': self.fc1(x_1),
        }

        x_4 = case2.get(self.option_list[3])
        x_4 = torch.relu(x_4)
        x_4 = F.dropout(x_4, p=self.dropout_frac)

        output = torch.cat((x_2, x_3, x_4), dim=1)
        output = self.fc_out(output)
        output = torch.sigmoid(output)

        return output

class MyGraphNetwork0011(nn.Module):
    option_list = None
    def __init__(self, in_features, out_features,hidden_dim=5, cheb_order=2, dropout=0.1):
        super(MyGraphNetwork0011, self).__init__()
        self.gcn = GCNConv(in_features, hidden_dim).to(device1)
        self.gat = GATConv(in_features, hidden_dim).to(device1)
        self.sage = SAGEConv(in_features, hidden_dim).to(device1)
        self.gin = GINConv(nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU())).to(device1)
        self.cheb = ChebConv(in_features, hidden_dim, cheb_order).to(device1)
        self.arma = ARMAConv(in_features, hidden_dim).to(device1)
        self.graph = GraphConv(in_features, hidden_dim).to(device1)
        self.skip = nn.Linear(in_features, hidden_dim).to(device1)
        self.fc = nn.Linear(in_features, hidden_dim).to(device1)

        self.gcn1 = GCNConv(hidden_dim, hidden_dim).to(device1)
        self.gat1 = GATConv(hidden_dim, hidden_dim).to(device1)
        self.sage1 = SAGEConv(hidden_dim, hidden_dim).to(device1)
        self.gin1 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())).to(device1)
        self.cheb1 = ChebConv(hidden_dim, hidden_dim, cheb_order).to(device1)
        self.arma1 = ARMAConv(hidden_dim, hidden_dim).to(device1)
        self.graph1 = GraphConv(hidden_dim, hidden_dim).to(device1)
        self.skip1 = nn.Linear(hidden_dim, hidden_dim).to(device1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim).to(device1)
        self.dropout_frac = dropout

        self.fc_out = nn.Linear(hidden_dim * 3, out_features).to(device1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        case = {
            'gcn': self.gcn(x, edge_index),
            'gat': self.gat(x, edge_index),
            'sage': self.sage(x, edge_index),
            'gin': self.gin(x, edge_index),
            'cheb': self.cheb(x, edge_index),
            'arma': self.arma(x, edge_index),
            'graph': self.graph(x, edge_index),
            'skip': self.skip(x),
            'fc': self.fc(x),
        }

        x_1 = case.get(self.option_list[0])
        x_1 = torch.relu(x_1)
        x_1 = F.dropout(x_1, p=self.dropout_frac)

        case2 = {
            'gcn': self.gcn1(x_1, edge_index),
            'gat': self.gat1(x_1, edge_index),
            'sage': self.sage1(x_1, edge_index),
            'gin': self.gin1(x_1, edge_index),
            'cheb': self.cheb1(x_1, edge_index),
            'arma': self.arma1(x_1, edge_index),
            'graph': self.graph1(x_1, edge_index),
            'skip': self.skip1(x_1),
            'fc': self.fc1(x_1),
        }
        x_2 = case.get(self.option_list[1])
        x_2 = torch.relu(x_2)
        x_2 = F.dropout(x_2, p=self.dropout_frac)

        x_3 = case2.get(self.option_list[2])
        x_3 = torch.relu(x_3)
        x_3 = F.dropout(x_3, p=self.dropout_frac)

        x_4 = case2.get(self.option_list[3])
        x_4 = torch.relu(x_4)
        x_4 = F.dropout(x_4, p=self.dropout_frac)

        output = torch.cat((x_2, x_3, x_4), dim=1)
        output = self.fc_out(output)
        output = torch.sigmoid(output)

        return output

class MyGraphNetwork0012(nn.Module):
    option_list = None
    def __init__(self, in_features, out_features, hidden_dim=5, cheb_order=2, dropout=0.1):
        super(MyGraphNetwork0012, self).__init__()
        self.gcn = GCNConv(in_features, hidden_dim).to(device1)
        self.gat = GATConv(in_features, hidden_dim).to(device1)
        self.sage = SAGEConv(in_features, hidden_dim).to(device1)
        self.gin = GINConv(nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU())).to(device1)
        self.cheb = ChebConv(in_features, hidden_dim, cheb_order).to(device1)
        self.arma = ARMAConv(in_features, hidden_dim).to(device1)
        self.graph = GraphConv(in_features, hidden_dim).to(device1)
        self.skip = nn.Linear(in_features, hidden_dim).to(device1)
        self.fc = nn.Linear(in_features, hidden_dim).to(device1)

        self.gcn1 = GCNConv(hidden_dim, hidden_dim).to(device1)
        self.gat1 = GATConv(hidden_dim, hidden_dim).to(device1)
        self.sage1 = SAGEConv(hidden_dim, hidden_dim).to(device1)
        self.gin1 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())).to(device1)
        self.cheb1 = ChebConv(hidden_dim, hidden_dim, cheb_order).to(device1)
        self.arma1 = ARMAConv(hidden_dim, hidden_dim).to(device1)
        self.graph1 = GraphConv(hidden_dim, hidden_dim).to(device1)
        self.skip1 = nn.Linear(hidden_dim, hidden_dim).to(device1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim).to(device1)
        self.dropout_frac = dropout

        self.fc_out = nn.Linear(hidden_dim * 2, out_features).to(device1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        case = {
            'gcn': self.gcn(x, edge_index),
            'gat': self.gat(x, edge_index),
            'sage': self.sage(x, edge_index),
            'gin': self.gin(x, edge_index),
            'cheb': self.cheb(x, edge_index),
            'arma': self.arma(x, edge_index),
            'graph': self.graph(x, edge_index),
            'skip': self.skip(x),
            'fc': self.fc(x),
        }

        x_1 = case.get(self.option_list[0])
        x_1 = torch.relu(x_1)
        x_1 = F.dropout(x_1, p=self.dropout_frac)

        case2 = {
            'gcn': self.gcn1(x_1, edge_index),
            'gat': self.gat1(x_1, edge_index),
            'sage': self.sage1(x_1, edge_index),
            'gin': self.gin1(x_1, edge_index),
            'cheb': self.cheb1(x_1, edge_index),
            'arma': self.arma1(x_1, edge_index),
            'graph': self.graph1(x_1, edge_index),
            'skip': self.skip1(x_1),
            'fc': self.fc1(x_1),
        }
        x_2 = case.get(self.option_list[1])
        x_2 = torch.relu(x_2)
        x_2 = F.dropout(x_2, p=self.dropout_frac)

        case3 = {
            'gcn': self.gcn1(x_2, edge_index),
            'gat': self.gat1(x_2, edge_index),
            'sage': self.sage1(x_2, edge_index),
            'gin': self.gin1(x_2, edge_index),
            'cheb': self.cheb1(x_2, edge_index),
            'arma': self.arma1(x_2, edge_index),
            'graph': self.graph1(x_2, edge_index),
            'skip': self.skip1(x_2),
            'fc': self.fc1(x_2),
        }

        x_3 = case2.get(self.option_list[2])
        x_3 = torch.relu(x_3)
        x_3 = F.dropout(x_3, p=self.dropout_frac)

        x_4 = case3.get(self.option_list[3])
        x_4 = torch.relu(x_4)
        x_4 = F.dropout(x_4, p=self.dropout_frac)

        output = torch.cat((x_3, x_4), dim=1)
        output = self.fc_out(output)
        output = torch.sigmoid(output)

        return output

class MyGraphNetwork0013(nn.Module):
    option_list = None
    def __init__(self, in_features, out_features, hidden_dim=5, cheb_order=2, dropout=0.1):
        super(MyGraphNetwork0013, self).__init__()
        self.gcn = GCNConv(in_features, hidden_dim).to(device1)
        self.gat = GATConv(in_features, hidden_dim).to(device1)
        self.sage = SAGEConv(in_features, hidden_dim).to(device1)
        self.gin = GINConv(nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU())).to(device1)
        self.cheb = ChebConv(in_features, hidden_dim, cheb_order).to(device1)
        self.arma = ARMAConv(in_features, hidden_dim).to(device1)
        self.graph = GraphConv(in_features, hidden_dim).to(device1)
        self.skip = nn.Linear(in_features, hidden_dim).to(device1)
        self.fc = nn.Linear(in_features, hidden_dim).to(device1)

        self.gcn1 = GCNConv(hidden_dim, hidden_dim).to(device1)
        self.gat1 = GATConv(hidden_dim, hidden_dim).to(device1)
        self.sage1 = SAGEConv(hidden_dim, hidden_dim).to(device1)
        self.gin1 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())).to(device1)
        self.cheb1 = ChebConv(hidden_dim, hidden_dim, cheb_order).to(device1)
        self.arma1 = ARMAConv(hidden_dim, hidden_dim).to(device1)
        self.graph1 = GraphConv(hidden_dim, hidden_dim).to(device1)
        self.skip1 = nn.Linear(hidden_dim, hidden_dim).to(device1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim).to(device1)
        self.dropout_frac = dropout

        self.fc_out = nn.Linear(hidden_dim * 2, out_features).to(device1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        case = {
            'gcn': self.gcn(x, edge_index),
            'gat': self.gat(x, edge_index),
            'sage': self.sage(x, edge_index),
            'gin': self.gin(x, edge_index),
            'cheb': self.cheb(x, edge_index),
            'arma': self.arma(x, edge_index),
            'graph': self.graph(x, edge_index),
            'skip': self.skip(x),
            'fc': self.fc(x),
        }

        x_1 = case.get(self.option_list[0])
        x_1 = torch.relu(x_1)
        x_1 = F.dropout(x_1, p=self.dropout_frac)

        case2 = {
            'gcn': self.gcn1(x_1, edge_index),
            'gat': self.gat1(x_1, edge_index),
            'sage': self.sage1(x_1, edge_index),
            'gin': self.gin1(x_1, edge_index),
            'cheb': self.cheb1(x_1, edge_index),
            'arma': self.arma1(x_1, edge_index),
            'graph': self.graph1(x_1, edge_index),
            'skip': self.skip1(x_1),
            'fc': self.fc1(x_1),
        }
        x_2 = case.get(self.option_list[1])
        x_2 = torch.relu(x_2)
        x_2 = F.dropout(x_2, p=self.dropout_frac)

        x_3 = case2.get(self.option_list[2])
        x_3 = torch.relu(x_3)
        x_3 = F.dropout(x_3, p=self.dropout_frac)

        case3 = {
            'gcn': self.gcn1(x_3, edge_index),
            'gat': self.gat1(x_3, edge_index),
            'sage': self.sage1(x_3, edge_index),
            'gin': self.gin1(x_3, edge_index),
            'cheb': self.cheb1(x_3, edge_index),
            'arma': self.arma1(x_3, edge_index),
            'graph': self.graph1(x_3, edge_index),
            'skip': self.skip1(x_3),
            'fc': self.fc1(x_3),
        }

        x_4 = case3.get(self.option_list[3])
        x_4 = torch.relu(x_4)
        x_4 = F.dropout(x_4, p=self.dropout_frac)

        output = torch.cat((x_2, x_4), dim=1)
        output = self.fc_out(output)
        output = torch.sigmoid(output)

        return output

class MyGraphNetwork0111(nn.Module):
    option_list = None
    def __init__(self, in_features, out_features, hidden_dim=5, cheb_order=2, dropout=0.1):
        super(MyGraphNetwork0111, self).__init__()
        self.gcn = GCNConv(in_features, hidden_dim).to(device1)
        self.gat = GATConv(in_features, hidden_dim).to(device1)
        self.sage = SAGEConv(in_features, hidden_dim).to(device1)
        self.gin = GINConv(nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU())).to(device1)
        self.cheb = ChebConv(in_features, hidden_dim, cheb_order).to(device1)
        self.arma = ARMAConv(in_features, hidden_dim).to(device1)
        self.graph = GraphConv(in_features, hidden_dim).to(device1)
        self.skip = nn.Linear(in_features, hidden_dim).to(device1)
        self.fc = nn.Linear(in_features, hidden_dim).to(device1)

        self.gcn1 = GCNConv(hidden_dim, hidden_dim).to(device1)
        self.gat1 = GATConv(hidden_dim, hidden_dim).to(device1)
        self.sage1 = SAGEConv(hidden_dim, hidden_dim).to(device1)
        self.gin1 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())).to(device1)
        self.cheb1 = ChebConv(hidden_dim, hidden_dim, cheb_order).to(device1)
        self.arma1 = ARMAConv(hidden_dim, hidden_dim).to(device1)
        self.graph1 = GraphConv(hidden_dim, hidden_dim).to(device1)
        self.skip1 = nn.Linear(hidden_dim, hidden_dim).to(device1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim).to(device1)
        self.dropout_frac = dropout

        self.fc_out = nn.Linear(hidden_dim * 3, out_features).to(device1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        case = {
            'gcn': self.gcn(x, edge_index),
            'gat': self.gat(x, edge_index),
            'sage': self.sage(x, edge_index),
            'gin': self.gin(x, edge_index),
            'cheb': self.cheb(x, edge_index),
            'arma': self.arma(x, edge_index),
            'graph': self.graph(x, edge_index),
            'skip': self.skip(x),
            'fc': self.fc(x),
        }

        x_1 = case.get(self.option_list[0])
        x_1 = torch.relu(x_1)
        x_1 = F.dropout(x_1, p=self.dropout_frac)

        case2 = {
            'gcn': self.gcn1(x_1, edge_index),
            'gat': self.gat1(x_1, edge_index),
            'sage': self.sage1(x_1, edge_index),
            'gin': self.gin1(x_1, edge_index),
            'cheb': self.cheb1(x_1, edge_index),
            'arma': self.arma1(x_1, edge_index),
            'graph': self.graph1(x_1, edge_index),
            'skip': self.skip1(x_1),
            'fc': self.fc1(x_1),
        }
        x_2 = case2.get(self.option_list[1])
        x_2 = torch.relu(x_2)
        x_2 = F.dropout(x_2, p=self.dropout_frac)

        x_3 = case2.get(self.option_list[2])
        x_3 = torch.relu(x_3)
        x_3 = F.dropout(x_3, p=self.dropout_frac)

        x_4 = case2.get(self.option_list[3])
        x_4 = torch.relu(x_4)
        x_4 = F.dropout(x_4, p=self.dropout_frac)

        output = torch.cat((x_2, x_3, x_4), dim=1)
        output = self.fc_out(output)
        output = torch.sigmoid(output)

        return output

class MyGraphNetwork0112(nn.Module):
    option_list = None
    def __init__(self, in_features, out_features, hidden_dim=5, cheb_order=2, dropout=0.1):
        super(MyGraphNetwork0112, self).__init__()
        self.gcn = GCNConv(in_features, hidden_dim).to(device1)
        self.gat = GATConv(in_features, hidden_dim).to(device1)
        self.sage = SAGEConv(in_features, hidden_dim).to(device1)
        self.gin = GINConv(nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU())).to(device1)
        self.cheb = ChebConv(in_features, hidden_dim, cheb_order).to(device1)
        self.arma = ARMAConv(in_features, hidden_dim).to(device1)
        self.graph = GraphConv(in_features, hidden_dim).to(device1)
        self.skip = nn.Linear(in_features, hidden_dim).to(device1)
        self.fc = nn.Linear(in_features, hidden_dim).to(device1)

        self.gcn1 = GCNConv(hidden_dim, hidden_dim).to(device1)
        self.gat1 = GATConv(hidden_dim, hidden_dim).to(device1)
        self.sage1 = SAGEConv(hidden_dim, hidden_dim).to(device1)
        self.gin1 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())).to(device1)
        self.cheb1 = ChebConv(hidden_dim, hidden_dim, cheb_order).to(device1)
        self.arma1 = ARMAConv(hidden_dim, hidden_dim).to(device1)
        self.graph1 = GraphConv(hidden_dim, hidden_dim).to(device1)
        self.skip1 = nn.Linear(hidden_dim, hidden_dim).to(device1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim).to(device1)
        self.dropout_frac = dropout

        self.fc_out = nn.Linear(hidden_dim * 2, out_features).to(device1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        case = {
            'gcn': self.gcn(x, edge_index),
            'gat': self.gat(x, edge_index),
            'sage': self.sage(x, edge_index),
            'gin': self.gin(x, edge_index),
            'cheb': self.cheb(x, edge_index),
            'arma': self.arma(x, edge_index),
            'graph': self.graph(x, edge_index),
            'skip': self.skip(x),
            'fc': self.fc(x),
        }

        x_1 = case.get(self.option_list[0])
        x_1 = torch.relu(x_1)
        x_1 = F.dropout(x_1, p=self.dropout_frac)

        case2 = {
            'gcn': self.gcn1(x_1, edge_index),
            'gat': self.gat1(x_1, edge_index),
            'sage': self.sage1(x_1, edge_index),
            'gin': self.gin1(x_1, edge_index),
            'cheb': self.cheb1(x_1, edge_index),
            'arma': self.arma1(x_1, edge_index),
            'graph': self.graph1(x_1, edge_index),
            'skip': self.skip1(x_1),
            'fc': self.fc1(x_1),
        }
        x_2 = case2.get(self.option_list[1])
        x_2 = torch.relu(x_2)
        x_2 = F.dropout(x_2, p=self.dropout_frac)

        case3 = {
            'gcn': self.gcn1(x_2, edge_index),
            'gat': self.gat1(x_2, edge_index),
            'sage': self.sage1(x_2, edge_index),
            'gin': self.gin1(x_2, edge_index),
            'cheb': self.cheb1(x_2, edge_index),
            'arma': self.arma1(x_2, edge_index),
            'graph': self.graph1(x_2, edge_index),
            'skip': self.skip1(x_2),
            'fc': self.fc1(x_2),
        }
        x_3 = case2.get(self.option_list[2])
        x_3 = torch.relu(x_3)
        x_3 = F.dropout(x_3, p=self.dropout_frac)

        x_4 = case3.get(self.option_list[3])
        x_4 = torch.relu(x_4)
        x_4 = F.dropout(x_4, p=self.dropout_frac)

        output = torch.cat((x_3, x_4), dim=1)
        output = self.fc_out(output)
        output = torch.sigmoid(output)

        return output

class MyGraphNetwork0122(nn.Module):
    option_list = None
    def __init__(self, in_features, out_features, hidden_dim=5, cheb_order=2, dropout=0.1):
        super(MyGraphNetwork0122, self).__init__()
        self.gcn = GCNConv(in_features, hidden_dim).to(device1)
        self.gat = GATConv(in_features, hidden_dim).to(device1)
        self.sage = SAGEConv(in_features, hidden_dim).to(device1)
        self.gin = GINConv(nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU())).to(device1)
        self.cheb = ChebConv(in_features, hidden_dim, cheb_order).to(device1)
        self.arma = ARMAConv(in_features, hidden_dim).to(device1)
        self.graph = GraphConv(in_features, hidden_dim).to(device1)
        self.skip = nn.Linear(in_features, hidden_dim).to(device1)
        self.fc = nn.Linear(in_features, hidden_dim).to(device1)

        self.gcn1 = GCNConv(hidden_dim, hidden_dim).to(device1)
        self.gat1 = GATConv(hidden_dim, hidden_dim).to(device1)
        self.sage1 = SAGEConv(hidden_dim, hidden_dim).to(device1)
        self.gin1 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())).to(device1)
        self.cheb1 = ChebConv(hidden_dim, hidden_dim, cheb_order).to(device1)
        self.arma1 = ARMAConv(hidden_dim, hidden_dim).to(device1)
        self.graph1 = GraphConv(hidden_dim, hidden_dim).to(device1)
        self.skip1 = nn.Linear(hidden_dim, hidden_dim).to(device1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim).to(device1)
        self.dropout_frac = dropout

        self.fc_out = nn.Linear(hidden_dim * 2, out_features).to(device1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        case = {
            'gcn': self.gcn(x, edge_index),
            'gat': self.gat(x, edge_index),
            'sage': self.sage(x, edge_index),
            'gin': self.gin(x, edge_index),
            'cheb': self.cheb(x, edge_index),
            'arma': self.arma(x, edge_index),
            'graph': self.graph(x, edge_index),
            'skip': self.skip(x),
            'fc': self.fc(x),
        }

        x_1 = case.get(self.option_list[0])
        x_1 = torch.relu(x_1)
        x_1 = F.dropout(x_1, p=self.dropout_frac)

        case2 = {
            'gcn': self.gcn1(x_1, edge_index),
            'gat': self.gat1(x_1, edge_index),
            'sage': self.sage1(x_1, edge_index),
            'gin': self.gin1(x_1, edge_index),
            'cheb': self.cheb1(x_1, edge_index),
            'arma': self.arma1(x_1, edge_index),
            'graph': self.graph1(x_1, edge_index),
            'skip': self.skip1(x_1),
            'fc': self.fc1(x_1),
        }
        x_2 = case2.get(self.option_list[1])
        x_2 = torch.relu(x_2)
        x_2 = F.dropout(x_2, p=self.dropout_frac)

        case3 = {
            'gcn': self.gcn1(x_2, edge_index),
            'gat': self.gat1(x_2, edge_index),
            'sage': self.sage1(x_2, edge_index),
            'gin': self.gin1(x_2, edge_index),
            'cheb': self.cheb1(x_2, edge_index),
            'arma': self.arma1(x_2, edge_index),
            'graph': self.graph1(x_2, edge_index),
            'skip': self.skip1(x_2),
            'fc': self.fc1(x_2),
        }
        x_3 = case3.get(self.option_list[2])
        x_3 = torch.relu(x_3)
        x_3 = F.dropout(x_3, p=self.dropout_frac)

        x_4 = case3.get(self.option_list[3])
        x_4 = torch.relu(x_4)
        x_4 = F.dropout(x_4, p=self.dropout_frac)

        output = torch.cat((x_3, x_4), dim=1)
        output = self.fc_out(output)
        output = torch.sigmoid(output)

        return output

class MyGraphNetwork0123(nn.Module):
    option_list = None
    def __init__(self, in_features, out_features, hidden_dim=5, cheb_order=2, dropout=0.1):
        super(MyGraphNetwork0123, self).__init__()
        self.gcn = GCNConv(in_features, hidden_dim).to(device1)
        self.gat = GATConv(in_features, hidden_dim).to(device1)
        self.sage = SAGEConv(in_features, hidden_dim).to(device1)
        self.gin = GINConv(nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU())).to(device1)
        self.cheb = ChebConv(in_features, hidden_dim, cheb_order).to(device1)
        self.arma = ARMAConv(in_features, hidden_dim).to(device1)
        self.graph = GraphConv(in_features, hidden_dim).to(device1)
        self.skip = nn.Linear(in_features, hidden_dim).to(device1)
        self.fc = nn.Linear(in_features, hidden_dim).to(device1)

        self.gcn1 = GCNConv(hidden_dim, hidden_dim).to(device1)
        self.gat1 = GATConv(hidden_dim, hidden_dim).to(device1)
        self.sage1 = SAGEConv(hidden_dim, hidden_dim).to(device1)
        self.gin1 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())).to(device1)
        self.cheb1 = ChebConv(hidden_dim, hidden_dim, cheb_order).to(device1)
        self.arma1 = ARMAConv(hidden_dim, hidden_dim).to(device1)
        self.graph1 = GraphConv(hidden_dim, hidden_dim).to(device1)
        self.skip1 = nn.Linear(hidden_dim, hidden_dim).to(device1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim).to(device1)
        self.dropout_frac = dropout

        self.fc_out = nn.Linear(hidden_dim, out_features).to(device1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        case = {
            'gcn': self.gcn(x, edge_index),
            'gat': self.gat(x, edge_index),
            'sage': self.sage(x, edge_index),
            'gin': self.gin(x, edge_index),
            'cheb': self.cheb(x, edge_index),
            'arma': self.arma(x, edge_index),
            'graph': self.graph(x, edge_index),
            'skip': self.skip(x),
            'fc': self.fc(x),
        }

        x_1 = case.get(self.option_list[0])
        x_1 = torch.relu(x_1)
        x_1 = F.dropout(x_1, p=self.dropout_frac)

        case2 = {
            'gcn': self.gcn1(x_1, edge_index),
            'gat': self.gat1(x_1, edge_index),
            'sage': self.sage1(x_1, edge_index),
            'gin': self.gin1(x_1, edge_index),
            'cheb': self.cheb1(x_1, edge_index),
            'arma': self.arma1(x_1, edge_index),
            'graph': self.graph1(x_1, edge_index),
            'skip': self.skip1(x_1),
            'fc': self.fc1(x_1),
        }
        x_2 = case2.get(self.option_list[1])
        x_2 = torch.relu(x_2)
        x_2 = F.dropout(x_2, p=self.dropout_frac)

        case3 = {
            'gcn': self.gcn1(x_2, edge_index),
            'gat': self.gat1(x_2, edge_index),
            'sage': self.sage1(x_2, edge_index),
            'gin': self.gin1(x_2, edge_index),
            'cheb': self.cheb1(x_2, edge_index),
            'arma': self.arma1(x_2, edge_index),
            'graph': self.graph1(x_2, edge_index),
            'skip': self.skip1(x_2),
            'fc': self.fc1(x_2),
        }
        x_3 = case3.get(self.option_list[2])
        x_3 = torch.relu(x_3)
        x_3 = F.dropout(x_3, p=self.dropout_frac)

        case4 = {
            'gcn': self.gcn1(x_3, edge_index),
            'gat': self.gat1(x_3, edge_index),
            'sage': self.sage1(x_3, edge_index),
            'gin': self.gin1(x_3, edge_index),
            'cheb': self.cheb1(x_3, edge_index),
            'arma': self.arma1(x_3, edge_index),
            'graph': self.graph1(x_3, edge_index),
            'skip': self.skip1(x_3),
            'fc': self.fc1(x_3),
        }
        x_4 = case4.get(self.option_list[3])
        x_4 = torch.relu(x_4)
        x_4 = F.dropout(x_4, p=self.dropout_frac)

        output = self.fc_out(x_4)
        output = torch.sigmoid(output)

        return output

def get_MyGNN(link):
    a = str(link)
    case = {
        '[0, 0, 0, 0]': MyGraphNetwork0000,
        '[0, 0, 0, 1]': MyGraphNetwork0001,
        '[0, 0, 1, 1]': MyGraphNetwork0011,
        '[0, 0, 1, 2]': MyGraphNetwork0012,
        '[0, 0, 1, 3]': MyGraphNetwork0013,
        '[0, 1, 1, 1]': MyGraphNetwork0111,
        '[0, 1, 1, 2]': MyGraphNetwork0112,
        '[0, 1, 2, 2]': MyGraphNetwork0122,
        '[0, 1, 2, 3]': MyGraphNetwork0123,
    }
    Mygnn = case.get(a)
    return Mygnn


def get_acc_list(link, all_egdes, option_list):
    gnn_list = option_list
    all_best_result = []
    acc_list = []
    for sublist in gnn_list:
        model = get_MyGNN(link)
        model.option_list = sublist
        # model_class, file_lock = args
        print(f'Running experiment for model:{model.option_list}')
        print("begin coding")

        IterNum = 1

        dim_embedding = 369
        in_features = dim_embedding

        print("G14 dataset")
        data = open("../G14.txt", 'r')
        reader = csv.reader(data)
        allRows = [list(map(int, row[0].split())) for row in reader]  # allRows is a list of Graph
        allRows = allRows[1:]
        for i in range(len(allRows)):
            del allRows[i][2]
        allRows = list(map(tuple, allRows))

        edge_index, graph_dgl, G, n_nodes = get_edge_index(allRows)

        A = jnp.array(nx.to_numpy_array(G))
        Q = create_Q_matrix(G)
        Q = Q.to(device1)

        net, embed, optimizer = create_gnn.get_gnn_params(in_features, 1, n_nodes, model)

        cut_vals = []
        best_solutiuon_dict = {0: 0}

        for i in range(IterNum):
            print(i)
            print('Running GNN...')
            net, embed, optimizer = create_gnn.get_gnn_params(in_features, 1, n_nodes, model)
            gnn_start = time()

            net, epoch, final_bitstring, best_bitstring, losses, epochs = create_gnn.run_gnn_training_GPT4GNAS(
                embed,
                graph_dgl,
                Q, net,
                optimizer,
                edge_index)

            gnn_time = time() - gnn_start
            bitstring_list = list(best_bitstring)

            best_bitstring = best_bitstring.type(dtype)
            best_bitstring = best_bitstring.to(device1)
            # q_torch_long = Q.type(torch.LongTensor)
            q_torch_long = Q.type(dtype)
            q_torch_long = q_torch_long.to(device1)

            cut_value_from_training = -(best_bitstring.T @ q_torch_long @ best_bitstring)
            cut_vals.append(cut_value_from_training)

            if cut_value_from_training > list(best_solutiuon_dict.keys())[0]:
                best_solutiuon_dict[cut_value_from_training] = best_solutiuon_dict.pop(
                    list(best_solutiuon_dict.keys())[0])
                best_solutiuon_dict[cut_value_from_training] = bitstring_list
        print(f"the best result of:{model.option_list} ", max(cut_vals))
        result = max(cut_vals)
        result = float(result)
        all_best_result.append(result)
        acc = result / all_egdes
        acc_list.append(acc)
        with open("experiment.txt", "a") as file:
            file.write(str(model.option_list) + "     " + str(result) + "\n")

    print(all_best_result)
    print(acc_list)

    return acc_list
