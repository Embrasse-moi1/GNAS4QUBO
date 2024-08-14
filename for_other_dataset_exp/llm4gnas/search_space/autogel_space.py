from for_other_dataset_exp.llm4gnas.search_space.gnn_base import *
from for_other_dataset_exp.llm4gnas.register import model_factory
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import argparse
import sys
from collections import defaultdict as ddict
from typing import Optional
from torch import Tensor
from torch_geometric.utils import scatter
from torch_geometric.data import Data


class Autogel(GNNBase):
    def __init__(self, desc, config):
        super(Autogel, self).__init__(desc, config)
        self.config = config
        self.auto_model = self.get_gnn(desc, config.in_dim, config.out_dim)
        if config.task_name == 'NodeClassification':
            self.readout = NodeClassificationHead(config)
        elif config.task_name == 'CO_problem':
            self.readout = CO_problem(config)
        elif config.task_name == 'GraphClassification':
            self.readout = GraphClassification(config)

    def get_gnn(self, oper_str, in_features, out_features, gpu=0):
        if self.config.task_name == "LinkPredict":
            op2z_mapping = {
                'agg': {'sum': [1.0, 0.0, 0.0], 'mean': [0.0, 1.0, 0.0], 'max': [0.0, 0.0, 1.0]},
                'combine': {'sum': [1.0, 0.0], 'concat': [0.0, 1.0]},
                'act': {'relu': [1.0, 0.0], 'prelu': [0.0, 1.0]},
                'layer_connect': {'stack': [1.0, 0.0, 0.0], 'skip_sum': [0.0, 1.0, 0.0], 'skip_cat': [0.0, 0.0, 1.0]},
                'layer_agg': {'none': [1.0, 0.0, 0.0], 'concat': [0.0, 1.0, 0.0], 'max_pooling': [0.0, 0.0, 1.0]},
                'pool': {'sum': [1.0, 0.0, 0.0], 'max': [0.0, 1.0, 0.0],
                         'concat': [0.0, 0.0, 1.0]}
            }
        elif self.config.task_name in ["NodeClassification", "GraphClassification"]:
            op2z_mapping = {
                'agg': {'sum': [1.0, 0.0, 0.0], 'mean': [0.0, 1.0, 0.0], 'max': [0.0, 0.0, 1.0]},
                'combine': {'sum': [1.0, 0.0], 'concat': [0.0, 1.0]},
                'act': {'relu': [1.0, 0.0], 'prelu': [0.0, 1.0]},
                'layer_connect': {'stack': [1.0, 0.0, 0.0], 'skip_sum': [0.0, 1.0, 0.0], 'skip_cat': [0.0, 0.0, 1.0]},
                'layer_agg': {'none': [1.0, 0.0, 0.0], 'concat': [0.0, 1.0, 0.0], 'max_pooling': [0.0, 0.0, 1.0]},
                'pool': {'global_add_pool': [1.0, 0.0, 0.0], 'global_mean_pool': [0.0, 1.0, 0.0],
                         'global_max_pool': [0.0, 0.0, 1.0]}
            }
        model_str_list = oper_str.split(';')
        layer_str = {'agg': ['', ''],
                     'combine': ['', ''],
                     'act': ['', ''],
                     'layer_connect': ['', ''],
                     'layer_agg': ['']}
        layer_str['layer_agg'][0] = model_str_list[2].split('layer_agg:')[1]
        for j in range(2):
            layer_str['agg'][j] = model_str_list[j].split('agg:')[1].split(',')[0]
            layer_str['combine'][j] = model_str_list[j].split('combine:')[1].split(',')[0]
            layer_str['act'][j] = model_str_list[j].split('act:')[1].split(',')[0]
            layer_str['layer_connect'][j] = model_str_list[j].split('layer_connect:')[1].split('}')[0]
        # {'agg': ['sum', 'sum'], 'combine': ['concat', 'concat'], 'act': ['prelu', 'relu'], 'layer_connect': ['skip_cat', 'stack'], 'layer_agg': ['concat']}
        # 获得layer_str中各个层次中的值
        if self.config.task_name == "LinkPredict":
            model = autogel_getmodel_lp(in_features, out_features, gpu=gpu, config=self.config)
        else:
            model = autogel_getmodel(in_features, out_features, gpu=gpu, seed=10, config=self.config)

        model.searched_arch_op = layer_str
        for key in model.searched_arch_op.keys():
            # model.searched_arch_z[key] = [op2z_mapping[key][k] for k in model.searched_arch_op[key]]
            if key not in op2z_mapping:
                raise KeyError(f"Key '{key}' not found in op2z_mapping")
            for k in model.searched_arch_op[key]:
                if k not in op2z_mapping[key]:
                    print(f"Warning: Value '{k}' not found in op2z_mapping for key '{key}'")
                    first_key = next(iter(op2z_mapping[key]))
                    model.searched_arch_z[key] = [op2z_mapping[key][first_key][0]]
                else:
                    model.searched_arch_z[key] = [op2z_mapping[key][k][0]]
        model.searched_arch_z = dict(model.searched_arch_z)
        model.searched_arch_op = dict(model.searched_arch_op)
        return model

    def loss(self, data: Data = None, out=None, prob=None, Q=None):
        if self.config.task_name == 'CO_problem':
            if prob is not None and Q is not None:
                loss_ = self.readout.task_loss(prob, Q)
            else:
                raise ValueError("prob or Q is None")
        else:
            if data is None or out is None:
                raise ValueError("data or out is None")
            else:
                loss_ = self.readout.task_loss(data, out)
        return loss_

    def metric(self, data: Union[Data, Tuple] = None, out=None, gnn=None, maxcut=None, total_edges=None):
        if self.config.task_name == 'CO_problem':
            if maxcut is not None and total_edges is not None:
                metric_ = self.readout.task_metric(maxcut, total_edges)
            else:
                raise ValueError("for 'CO_problem', values for 'maxcut' and 'total_edges' are required.")
        elif self.config.task_name == "GraphClassification":
            if data is not None and gnn is not None:
                metric_ = self.readout.task_metric(data, gnn)
            else:
                raise ValueError("data or gnn is None")
        else:
            if data is None or out is None:
                raise ValueError("data or out is None")
            else:
                metric_ = self.readout.task_metric(data, out)
        return metric_

    def forward(self, data):
        x = self.auto_model.forward(data)
        return x


class Sum_AGG(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Sum_AGG, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        neighbor_info = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        return neighbor_info

    def massage(self, x_j, edge_index, size):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class Mean_AGG(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Mean_AGG, self).__init__(aggr='mean')

    def forward(self, x, edge_index):
        neighbor_info = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        return neighbor_info

    def massage(self, x_j, edge_index, size):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class Max_AGG(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Max_AGG, self).__init__(aggr='max')

    def forward(self, x, edge_index):
        neighbor_info = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        return neighbor_info

    def massage(self, x_j, edge_index, size):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class SearchSpace(object):
    def __init__(self, task_name):
        # self.gcnconv = ['GINConv', 'GCNConv', 'SAGEConv', 'GATConv']
        self.agg = ['sum', 'mean', 'max']
        self.combine = ['sum', 'concat']
        self.act = ['relu', 'prelu']
        self.layer_connect = ['stack', 'skip_sum', 'skip_cat']
        self.layer_agg = ['none', 'concat', 'max_pooling']
        # self.pool = ['sum', 'diff', 'hadamard', 'max', 'concat']
        # self.pool = ['sum']
        # self.pool = ['sum', 'max', 'concat']
        if task_name == "LinkPredict":
            self.pool = ['sum', 'max', 'concat']
        elif task_name in ["NodeClassification", "GraphClassification"]:
            self.pool = ['global_add_pool', 'global_mean_pool', 'global_max_pool']

        self.search_space = {'agg': self.agg,
                             'combine': self.combine,
                             'act': self.act,
                             'layer_connect': self.layer_connect,
                             'layer_agg': self.layer_agg,
                             'pool': self.pool}

        self.dims = list(self.search_space.keys())
        self.choices = []
        self.num_choices = {}
        for dim in self.dims:
            self.choices.append(self.search_space[dim])
            self.num_choices[dim] = len(self.search_space[dim])

    def get_search_space(self):
        return self.search_space


def global_add_pool(x: Tensor, batch: Optional[Tensor],
                    size: Optional[int] = None) -> Tensor:
    r"""Returns batch-wise graph-level-outputs by adding node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n.

    Functional method of the
    :class:`~torch_geometric.nn.aggr.SumAggregation` module.

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (torch.Tensor, optional): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node to a specific example.
        size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    dim = -1 if x.dim() == 1 else -2

    if batch is None:
        return x.sum(dim=dim, keepdim=x.dim() <= 2)
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=dim, dim_size=size, reduce='sum')


def global_mean_pool(x: Tensor, batch: Optional[Tensor],
                     size: Optional[int] = None) -> Tensor:
    r"""Returns batch-wise graph-level-outputs by averaging node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \mathbf{x}_n.

    Functional method of the
    :class:`~torch_geometric.nn.aggr.MeanAggregation` module.

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (torch.Tensor, optional): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node to a specific example.
        size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    dim = -1 if x.dim() == 1 else -2

    if batch is None:
        return x.mean(dim=dim, keepdim=x.dim() <= 2)
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=dim, dim_size=size, reduce='mean')


def global_max_pool(x: Tensor, batch: Optional[Tensor],
                    size: Optional[int] = None) -> Tensor:
    r"""Returns batch-wise graph-level-outputs by taking the channel-wise
    maximum across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \mathrm{max}_{n=1}^{N_i} \, \mathbf{x}_n.

    Functional method of the
    :class:`~torch_geometric.nn.aggr.MaxAggregation` module.

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (torch.Tensor, optional): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each element to a specific example.
        size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    dim = -1 if x.dim() == 1 else -2

    if batch is None:
        return x.max(dim=dim, keepdim=x.dim() <= 2)[0]
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=dim, dim_size=size, reduce='max')


class GNNModel(nn.Module):
    def get_device(self, args):
        gpu = args.gpu
        return torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')

    def sample_gumbel(self, shape, args, eps=1e-20):
        U = torch.rand(shape, device=self.device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, log_alpha, temperature, args):
        y = log_alpha + self.sample_gumbel(log_alpha.size(), args)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, log_alpha, temperature, args, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(log_alpha, temperature, args)

        if not hard:
            return y
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        # y_hard=y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y, y_hard

    def get_Z_hard(self, log_alpha, temperature, args):
        Z_hard = self.gumbel_softmax(log_alpha, temperature, args, hard=True)[1]
        return Z_hard

    #################################################################################
    def load_searchspace(self):
        self.SearchSpace = SearchSpace(self.args.task_name)
        self.search_space = self.SearchSpace.search_space
        self.num_choices = self.SearchSpace.num_choices
        self.dims = self.SearchSpace.dims
        self.temperature = 1

    #################################################################################
    def init_alpha(self):
        self.log_alpha_agg = nn.Parameter(torch.zeros((self.layers, self.num_choices['agg']),
                                                      device=self.device).normal_(mean=1, std=0.01).requires_grad_())
        self.log_alpha_combine = nn.Parameter(torch.zeros((self.layers, self.num_choices['combine']),
                                                          device=self.device).normal_(mean=1,
                                                                                      std=0.01).requires_grad_())
        self.log_alpha_act = nn.Parameter(torch.zeros((self.layers, self.num_choices['act']),
                                                      device=self.device).normal_(mean=1, std=0.01).requires_grad_())
        self.log_alpha_layer_connect = nn.Parameter(torch.zeros(self.layers, self.num_choices['layer_connect'],
                                                                device=self.device).normal_(mean=1,
                                                                                            std=0.01).requires_grad_())
        self.log_alpha_layer_agg = nn.Parameter(torch.zeros(1, self.num_choices['layer_agg'],
                                                            device=self.device).normal_(mean=1,
                                                                                        std=0.01).requires_grad_())
        self.log_alpha_pool = nn.Parameter(torch.zeros(1, self.num_choices['pool'],
                                                       device=self.device).normal_(mean=1, std=0.01).requires_grad_())

    #################################################################################
    def update_z_hard(self):
        # self.Z_gcnconv_hard = self.get_Z_hard(self.log_alpha_gcnconv, self.temperature, self.args)
        self.Z_agg_hard = self.get_Z_hard(self.log_alpha_agg, self.temperature, self.args)
        self.Z_combine_hard = self.get_Z_hard(self.log_alpha_combine, self.temperature, self.args)
        self.Z_act_hard = self.get_Z_hard(self.log_alpha_act, self.temperature, self.args)
        self.Z_layer_connect_hard = self.get_Z_hard(self.log_alpha_layer_connect, self.temperature, self.args)
        self.Z_layer_agg_hard = self.get_Z_hard(self.log_alpha_layer_agg, self.temperature, self.args)
        self.Z_pool_hard = self.get_Z_hard(self.log_alpha_pool, self.temperature, self.args)

        self.Z_hard_dict['agg'].append(self.Z_agg_hard.cpu().tolist())
        self.Z_hard_dict['combine'].append(self.Z_combine_hard.cpu().tolist())
        self.Z_hard_dict['act'].append(self.Z_act_hard.cpu().tolist())
        self.Z_hard_dict['layer_connect'].append(self.Z_layer_connect_hard.cpu().tolist())
        self.Z_hard_dict['layer_agg'].append(self.Z_layer_agg_hard.cpu().tolist())
        self.Z_hard_dict['pool'].append(self.Z_pool_hard.cpu().tolist())
        self.Z_hard_dict = dict(self.Z_hard_dict)

    def derive_arch(self):
        for key in self.search_space.keys():
            self.searched_arch_z[key] = self.Z_hard_dict[key][self.max_step]
            self.searched_arch_op[key] = self.z2op(key, self.searched_arch_z[key])
        self.searched_arch_z = dict(self.searched_arch_z)
        self.searched_arch_op = dict(self.searched_arch_op)

        self.Z_agg_hard = torch.tensor(self.searched_arch_z['agg'], device=self.device)
        self.Z_combine_hard = torch.tensor(self.searched_arch_z['combine'], device=self.device)
        self.Z_act_hard = torch.tensor(self.searched_arch_z['act'], device=self.device)
        self.Z_layer_connect_hard = torch.tensor(self.searched_arch_z['layer_connect'], device=self.device)
        self.Z_layer_agg_hard = torch.tensor(self.searched_arch_z['layer_agg'], device=self.device)
        self.Z_pool_hard = torch.tensor(self.searched_arch_z['pool'], device=self.device)

    def z2op(self, key, z_hard):
        ops = []
        for i in range(len(z_hard)):
            index = z_hard[i].index(1)
            op = self.search_space[key][index]
            ops.append(op)
        return ops

    def load_agg(self):
        self.sum_agg = Sum_AGG(in_channels=self.hidden_features, out_channels=self.hidden_features)
        self.mean_agg = Mean_AGG(in_channels=self.hidden_features, out_channels=self.hidden_features)
        self.max_agg = Max_AGG(in_channels=self.hidden_features, out_channels=self.hidden_features)

    ###################################################################################################################
    ###################################################################################################################
    def __init__(self, layers, in_features, hidden_features, out_features, args, dropout=0.0):
        super(GNNModel, self).__init__()
        self.layers, self.in_features, self.hidden_features, self.out_features, self.args, = layers, in_features, hidden_features, out_features, args
        self.device = self.get_device(self.args)
        self.relu = nn.ReLU()
        self.prelu = nn.ModuleList([nn.PReLU() for i in range(layers)])
        self.dropout = nn.Dropout(p=dropout)

        self.preprocess = nn.Linear(in_features, hidden_features)
        self.linears = nn.ModuleList([nn.Linear(hidden_features, hidden_features) for i in range(layers)])
        self.linears_self = nn.ModuleList([nn.Linear(hidden_features, hidden_features) for i in range(layers)])
        self.combine_merger = nn.ModuleList([nn.Linear(2 * hidden_features, hidden_features) for i in range(layers)])
        self.layer_connect_merger = nn.ModuleList(
            [nn.Linear(2 * hidden_features, hidden_features) for i in range(layers)])
        self.layer_agg_merger = nn.Linear((layers + 1) * hidden_features, hidden_features)
        self.pool_merger = nn.Linear(2 * hidden_features, hidden_features)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_features) for i in range(layers)])
        self.feed_forward = FeedForwardNetwork(hidden_features, out_features)

        # self.init_superlayer()
        self.Z_hard_dict = ddict(list)
        self.searched_arch_z = ddict(list)
        self.searched_arch_op = ddict(list)
        self.load_searchspace()
        self.load_agg()
        self.init_alpha()
        self.update_z_hard()

        self.max_step = None
        self.best_metric_search = None
        self.best_metric_retrain = None
        self.args = args

    def forward(self, data):
        # x, batch, edge_index = data.x, data.batch, data.edge_index
        if self.args.task_name == "GraphClassification":
            batch = data
            x, batch, edge_index = batch.x, batch.batch, batch.edge_index
        else:
            batch = data
            x = batch.x
            edge_index = batch.edge_index
        try:
            batch = data.batch
        except AttributeError:
            batch = torch.zeros(data[0].num_nodes, dtype=torch.long)

        if data.num_node_features == 0:
            x = torch.ones((data.num_nodes, 1), device=self.device)

        x = self.preprocess(x)
        self.emb_list = [x]

        for i in range(self.layers):
            #######################################################################################################
            x_self, x_n = self.linears_self[i](x), self.linears[i](x)
            x_n = self.agg_trans(x_n, edge_index, self.Z_agg_hard[i].view(1, -1))
            x = self.combine_trans(i, x_self, x_n, self.Z_combine_hard[i].view(1, -1))
            #######################################################################################################
            # x = self.act(x)
            x = self.act_trans(i, x, self.Z_act_hard[i].view(1, -1))
            x = self.dropout(x)  # [n_nodes, mini_batch, input_dim]
            x = self.layer_norms[i](x)

            self.emb_list.append(x)
            x = self.layer_connect_trans(i + 1, self.Z_layer_connect_hard[i].view(1, -1))
        x = self.layer_agg_trans(self.Z_layer_agg_hard)
        self.emb_list = []

        if self.args.task_name == "GraphClassification":
            # x = self.get_minibatch_embeddings(x, batch)
            # x = global_add_pool(x, batch)
            # x = global_mean_pool(x, batch)
            # x = global_max_pool(x, batch)
            # x = global_sort_pool(x, batch, 1)
            x = self.global_pool_trans(x, batch, self.Z_pool_hard)
        if self.args.task_name == 'LinkPredict':
            x = self.get_minibatch_embeddings(x, data)
        x = self.feed_forward(x)
        return x

    def get_minibatch_embeddings(self, x, batch):
        device = x.device
        set_indices, batch, num_graphs = batch.set_indices, batch.batch, batch.num_graphs
        num_nodes = torch.eye(num_graphs, device=device)[batch].sum(dim=0)
        zero = torch.tensor([0], dtype=torch.long).to(device)
        index_bases = torch.cat([zero, torch.cumsum(num_nodes, dim=0, dtype=torch.long)[:-1]])
        index_bases = index_bases.unsqueeze(1).expand(-1, set_indices.size(-1))
        assert (index_bases.size(0) == set_indices.size(0))
        set_indices_batch = index_bases + set_indices
        # # print('set_indices shape', set_indices.shape, 'index_bases shape', index_bases.shape, 'x shape:', x.shape)
        # print(set_indices_batch.shape)
        # print(x.shape)
        x = x[set_indices_batch]  # shape [B, set_size, F], set_size=1, 2, or 3 for node, link and tri
        # print(x.shape)
        # x = self.pool(x)
        x = self.pool_trans(x, self.Z_pool_hard)
        return x

    def agg_trans(self, x_n, edge_index, z_hard):
        y = []
        for agg in [self.sum_agg, self.mean_agg, self.max_agg]:
            temp = agg(x_n, edge_index)
            y.append(temp)
        x_n = torch.stack(y, dim=0)
        x_n = torch.einsum('ij,jkl -> ikl', z_hard, x_n).squeeze(0)
        return x_n

    def combine_trans(self, cur_layer, x, x_n, z_hard):
        y = []
        for combine in self.search_space['combine']:
            temp = self.combine_map(cur_layer, x, x_n, combine)
            y.append(temp)
        x = torch.stack(y, dim=0)
        x = torch.einsum('ij,jkl -> ikl', z_hard, x).squeeze(0)
        return x

    def combine_map(self, cur_layer, x, x_n, combine):
        if combine == 'sum':
            x = x + x_n
        if combine == 'concat':
            x = torch.cat([x, x_n], axis=-1)
            x = self.combine_merger[cur_layer](x)
        return x

    def act_trans(self, cur_layer, x, z_hard):
        y = []
        for act in self.search_space['act']:
            temp = self.act_map(cur_layer, x, act)
            y.append(temp)
        x = torch.stack(y, dim=0)
        x = torch.einsum('ij,jkl -> ikl', z_hard, x).squeeze(0)
        return x

    def act_map(self, cur_layer, x, act):
        if act == 'relu':
            x = self.relu(x)
        if act == 'prelu':
            x = self.prelu[cur_layer](x)
        return x

    def layer_connect_trans(self, cur_layer, z_hard):
        y = []
        for layer_connect in self.search_space['layer_connect']:
            temp = self.layer_connect_map(cur_layer, layer_connect)
            y.append(temp)
        x = torch.stack(y, dim=0)
        x = torch.einsum('ij,jkl -> ikl', z_hard, x).squeeze(0)
        return x

    def layer_connect_map(self, cur_layer, layer_connect):
        if layer_connect == 'stack':
            x = self.emb_list[-1]
        if layer_connect == 'skip_sum':
            x = self.emb_list[-1] + self.emb_list[-2]
        if layer_connect == 'skip_cat':
            x = torch.cat([self.emb_list[-2], self.emb_list[-1]], dim=-1)
            x = self.layer_connect_merger[cur_layer - 1](x)
        return x

    def layer_agg_trans(self, z_hard):
        y = []
        for layer_agg in self.search_space['layer_agg']:
            #             temp = self.layer_agg_map(x, layer_agg)
            temp = self.layer_agg_map(layer_agg)
            y.append(temp)
        x = torch.stack(y, dim=0)
        x = torch.einsum('ij,jkl -> ikl', z_hard, x).squeeze(0)
        return x

    def layer_agg_map(self, layer_agg):
        if layer_agg == 'none':
            x = self.emb_list[-1]
        if layer_agg == 'max_pooling':
            # x = torch.stack(self.emb_list).to(self.device)
            x = torch.stack(self.emb_list)
            x = x.max(dim=0)[0]
        if layer_agg == 'concat':
            # x = torch.cat(self.emb_list, dim=-1).to(self.device)
            x = torch.cat(self.emb_list, dim=-1)
            x = self.layer_agg_merger(x)
        return x

    def global_pool_trans(self, x, batch, z_hard):
        y = []
        for pool in self.search_space['pool']:
            temp = self.global_pool_map(x, batch, pool)
            y.append(temp)
        x = torch.stack(y, dim=0)
        x = torch.einsum('ij,jkl -> ikl', z_hard, x).squeeze(0)
        return x

    def global_pool_map(self, x, batch, pool):
        if pool == 'global_add_pool':
            x = global_add_pool(x, batch)
        if pool == 'global_mean_pool':
            x = global_mean_pool(x, batch)
        if pool == 'global_max_pool':
            x = global_max_pool(x, batch)
        return x

    def pool_trans(self, x, z_hard):
        y = []
        for pool in self.search_space['pool']:
            temp = self.pool_map(x, pool)
            y.append(temp)
        x = torch.stack(y, dim=0)
        # print(z_hard.shape)
        # print(x.shape)
        x = torch.einsum('ij,jkl -> ikl', z_hard, x).squeeze(0)
        return x

    def pool_map(self, x, pool):
        if pool == 'sum':
            x = x.sum(dim=1)
        if pool == 'max':
            x = x.max(dim=1)[0]
        # if pool == 'diff':
        #     x_diff = torch.zeros_like(x[:, 0, :], device=x.device)
        #     for i, j in combinations(range(x.size(1)), 2):
        #         x_diff += torch.abs(x[:, i, :] - x[:, j, :])
        #         x = x_diff
        if pool == 'concat':
            x = x.view(x.shape[0], -1)
            x = self.pool_merger(x)
        return x

    def short_summary(self):
        return 'Model: Auto-GNN, #layers: {}, in_features: {}, hidden_features: {}, out_features: {}'.format(
            self.layers,
            self.in_features,
            self.hidden_features,
            self.out_features)


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_features, out_features, act=nn.ReLU(), dropout=0):
        super(FeedForwardNetwork, self).__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.layer1 = nn.Sequential(nn.Linear(in_features, in_features), self.act, self.dropout)
        # self.layer2 = nn.Sequential(nn.Linear(in_features, out_features), nn.LogSoftmax(dim=-1))
        self.layer2 = nn.Linear(in_features, out_features)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return x


def get_model(layers, in_features, out_features, args):
    model_name = args.model
    if model_name in ['Auto-GNN']:
        model = GNNModel(layers=layers, in_features=in_features, hidden_features=args.hid_dim,
                         out_features=out_features, args=args, dropout=args.dropout)
    else:
        return NotImplementedError
    return model


def autogel_getmodel(in_features, out_features, gpu=0, seed=10, config=None):
    # parser = argparse.ArgumentParser('Interface for Auto-GNN framework')
    # parser.add_argument('--train_ratio', type=float, default=0.6, help='ratio of the train against whole')
    # parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio of the val against whole')
    # parser.add_argument('--test_ratio', type=float, default=0.2, help='ratio of the test against whole')
    # parser.add_argument('--model', type=str, default='Auto-GNN', help='model to use')
    # parser.add_argument('--metric', type=str, default='acc', help='metric for evaluating performance',
    #                     choices=['acc', 'auc'])
    # parser.add_argument('--gpu', type=int, default=gpu, help='gpu id')
    # parser.add_argument('--directed', type=bool, default=False,
    #                     help='(Currently unavailable) whether to treat the graph as directed')
    # parser.add_argument('--parallel', default=False, action='store_true',
    #                     help='(Currently unavailable) whether to use multi cpu cores to prepare data')
    # parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use')
    #
    # parser.add_argument('--task', type=str, default='NodeClassification', help='type of task', choices=['NodeClassification', "GraphClassification", "LinkPredict"])
    # parser.add_argument('--seed', type=int, default=seed, help='seed to initialize all the random modules')
    # parser.add_argument('--clip', type=float, default=0.1, help='gradient clipping')
    # # general model and training setting
    # parser.add_argument('--dataset', type=str, default='Cora',
    #                     help='dataset name')  # choices=['PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'BZR', 'COX2', 'DD', 'ENZYMES', 'NCI1']
    #
    # parser.add_argument('--epoch', type=int, default=20, help='number of epochs to search')
    # parser.add_argument('--retrain_epoch', type=int, default=200, help='number of epochs to retrain')
    #
    # parser.add_argument('--layers', type=int, default=2, help='largest number of layers')
    # parser.add_argument('--hidden_features', type=int, default=128, help='hidden dimension')
    # parser.add_argument('--bs', type=int, default=128, help='minibatch size')
    # parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight')
    # parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    # # parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    # # logging & debug
    # parser.add_argument('--log_dir', type=str, default='./log/', help='log directory')
    #
    # parser.add_argument('--summary_file', type=str, default='result_summary.log',
    #                     help='brief summary of training result')
    # parser.add_argument('--debug', default=False, action='store_true',
    #                     help='whether to use debug mode')
    # parser.add_argument('--task_name', type=str, default='NodeClassification', help='log directory')
    # try:
    #     args = parser.parse_args()
    # except:
    #     parser.print_help()
    #     sys.exit(0)

    args = config

    model = get_model(layers=args.layers, in_features=in_features, out_features=out_features, args=args)
    return model


def autogel_getmodel_lp(in_features, out_features, gpu=0, config=None):
    """
    parser = argparse.ArgumentParser('Interface for Auto-GNN framework')
    # general model and training setting
    parser.add_argument('--gpu', type=int, default=gpu, help='gpu id')

    parser.add_argument('--dataset', type=str, default='celegans_small', help='dataset name') # currently relying on dataset to determine task
    parser.add_argument('--task', type=str, default='LinkPredict', help='type of task', choices=['NodeClassification', "GraphClassification", "LinkPredict"])
    parser.add_argument('--test_ratio', type=float, default=0.1, help='ratio of the test against whole')
    # parser.add_argument('--model', type=str, default='Auto-GNN', help='model to use', choices=['DE-GNN', 'GIN', 'GCN', 'GraphSAGE', 'GAT'])
    parser.add_argument('--model', type=str, default='Auto-GNN', help='model to use')
    parser.add_argument('--layers', type=int, default=2, help='largest number of layers')
    parser.add_argument('--hidden_features', type=int, default=100, help='hidden dimension')
    parser.add_argument('--metric', type=str, default='auc', help='metric for evaluating performance', choices=['acc', 'auc'])
    parser.add_argument('--seed', type=int, default=3, help='seed to initialize all the random modules')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    # parser.add_argument('--adj_norm', type=str, default='asym', help='how to normalize adj', choices=['asym', 'sym', 'None'])
    parser.add_argument('--data_usage', type=float, default=1.0, help='use partial dataset')
    parser.add_argument('--directed', type=bool, default=False, help='(Currently unavailable) whether to treat the graph as directed')
    parser.add_argument('--parallel', default=False, action='store_true',
                        help='(Currently unavailable) whether to use multi cpu cores to prepare data')

    # features and positional encoding
    parser.add_argument('--prop_depth', type=int, default=1, help='propagation depth (number of hops) for one layer')
    parser.add_argument('--use_degree', type=bool, default=True, help='whether to use node degree as the initial feature')
    parser.add_argument('--use_attributes', type=bool, default=False, help='whether to use node attributes as the initial feature')
    parser.add_argument('--feature', type=str, default='sp', help='distance encoding category: shortest path or random walk (landing probabilities)')  # sp (shortest path) or rw (random walk)
    parser.add_argument('--rw_depth', type=int, default=3, help='random walk steps')  # for random walk feature
    parser.add_argument('--max_sp', type=int, default=50, help='maximum distance to be encoded for shortest path feature')

    # model training
    parser.add_argument('--epoch', type=int, default=300, help='number of epochs to search')
    parser.add_argument('--retrain_epoch', type=int, default=300, help='number of epochs to retrain')
    parser.add_argument('--bs', type=int, default=64, help='minibatch size')
    # parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization weight')
    parser.add_argument('--clip', type=float, default=1, help='gradient clipping')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate')

    # simulation (valid only when dataset == 'simulation')
    parser.add_argument('--k', type=int, default=3, help='node degree (k) or synthetic k-regular graph')
    parser.add_argument('--n', nargs='*', help='a list of number of nodes in each connected k-regular subgraph')
    parser.add_argument('--N', type=int, default=1000, help='total number of nodes in simultation')
    parser.add_argument('--T', type=int, default=6, help='largest number of layers to be tested')

    # logging & debug
    # parser.add_argument('--log_dir', type=str, default='/export/data/zhili/PycharmProjects/NAS4GNN/homo/auto/v2/log/', help='log directory')  # sp (shortest path) or rw (random walk)
    parser.add_argument('--log_dir', type=str, default='./log/', help='log directory')  # sp (shortest path) or rw (random walk)

    parser.add_argument('--summary_file', type=str, default='result_summary.log', help='brief summary of training result')  # sp (shortest path) or rw (random walk)
    parser.add_argument('--debug', default=False, action='store_true',
                        help='whether to use debug mode')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    """

    args = config
    model = get_model(layers=args.layers, in_features=in_features, out_features=out_features,
                       args=args)
    return model

# model_factory["autogel_space"] = Autogel
