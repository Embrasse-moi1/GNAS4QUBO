#训练Gnn策略
from itertools import chain
from time import time
from co_corefunc import loss_func
import torch
import torch.nn as nn
from torch_geometric.data import Data
device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

out = 1000  #每一百次循环输出一次值 1000
prob_threshold = 0.5
number_epochs = 15000
tol = 1e-4
patience = 1000
learning_rate = 0.01
opt_params = {'lr': learning_rate}

#get embed, net, optimizer
def get_gnn_params(in_features, class_num, n_nodes, MyGraphNetwork):

    net = MyGraphNetwork(in_features, class_num)
    dim_embedding = in_features
    net = net.type(dtype).to(device1)
    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(dtype).to(device1)

    params = chain(net.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(params, **opt_params)

    return net, embed, optimizer

def run_gnn_training_GPT4GNAS(embed, dgl_graph, q_torch, net, optimizer, edge_index):
    inputs = embed.weight
    data = Data(x=inputs, edge_index=edge_index)
    prev_loss = 1.
    count = 0

    losses = []
    epochs = []

    best_bitstring = torch.zeros((dgl_graph.number_of_nodes(),)).type(q_torch.dtype).to(
        q_torch.device)  # 初始化全为0一个二进制张量，将图中每个节点关联一个二进制变量x
    best_loss = loss_func(best_bitstring.float(), q_torch)

    print("best_bitstring_shape", best_bitstring.shape)
    print("best_loss_shape", best_loss.shape)

    t_gnn_start = time()
    for epoch in range(number_epochs):
        probs = net(data)[:, 0]
        loss = loss_func(probs, q_torch)
        loss_ = loss.detach().item()

        bitstring = (probs.detach() >= prob_threshold) * 1
        if loss < best_loss:
            best_loss = loss
            best_bitstring = bitstring

        if epoch % out == 0:
            print(f'Epoch: {epoch}, Loss:{loss_}')
            losses.append(loss_)
            epochs.append(epoch)

        if(abs(loss_ - prev_loss) <= tol) | ((loss_ - prev_loss) > 0):
            count += 1
        else:
            count = 0

        if count >= patience:
            print(f'Stopping early on epoch {epoch}(patience: {patience})')
            break

        prev_loss = loss_

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    t_gnn = time() - t_gnn_start
    print(f'GNN training (n={dgl_graph.number_of_nodes()}) took {round(t_gnn, 3)}')
    print(f'GNN final continuous loss: {loss_}')
    print(f'GNN best continuous loss: {best_loss}')
    #print(best_bitstring)

    finial_bitstring = (probs.detach() >= prob_threshold) * 1

    return net, epoch, finial_bitstring, best_bitstring, losses, epochs
