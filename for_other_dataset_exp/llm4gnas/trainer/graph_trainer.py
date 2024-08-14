import torch
from torch import Tensor, scatter
from tqdm import tqdm
from typing import Union, Tuple
from torch_geometric.data import Data, Dataset
from for_other_dataset_exp.llm4gnas.register import model_factory
from for_other_dataset_exp.llm4gnas.utils.data import get_loader, get_optimizer
from for_other_dataset_exp.llm4gnas.trainer.trainer_base import TrainerBase
from for_other_dataset_exp.llm4gnas.search_space import GNNBase


class GraphClassificationTrainer(TrainerBase):
    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)
        if config.task_name != "GraphClassification":
            raise ValueError(f"Task name should be GraphClassification, but got {config.task_name}")
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def fit(self, dataset: Dataset, gnn: GNNBase, config: dict = None) -> GNNBase:
        config = self.config if config is None else config
        if config is None:
            raise ValueError("config is None.")

        if self.train_loader is None or self.val_loader is None or self.test_loader is None:
            self.train_loader, self.val_loader, self.test_loader = get_loader(dataset, config)

        self.optimizer = get_optimizer(gnn, config)
        gnn = gnn.to(self.device)
        for epoch in tqdm(range(1, config.epochs + 1)):
            gnn.train()
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                prediction = gnn(batch)
                # print(prediction)
                loss = gnn.loss(data=batch, out=prediction)
                # print(loss)
                loss.backward(retain_graph=True)
                self.optimizer.step()
        self.gnn = gnn
        return gnn

    def evaluate(self, dataset: Dataset, gnn: Union[GNNBase, None] = None) -> dict:
        gnn = self.gnn if gnn is None else gnn.to(self.device)
        if gnn is None and gnn is None:
            raise RuntimeError("GNN is not Given.")
        gnn.eval()
        dataloader = (self.train_loader, self.val_loader, self.test_loader)
        return gnn.metric(data=dataloader, gnn=gnn)

    def predict(self, dataset: Dataset, gnn: Union[GNNBase, None] = None) -> Tensor:
        dataset = dataset[0].to(self.device)
        # pred = []
        gnn = self.gnn if gnn is None else gnn
        if gnn is None and self.gnn is None:
            raise RuntimeError("GNN is not Given.")
        gnn.eval()
        return gnn(dataset)


model_factory["graph_trainer"] = GraphClassificationTrainer

if __name__ == "__main__":
    from torch_geometric.datasets import TUDataset
    from easydict import EasyDict as edict

    dataset_name = 'IMDB-BINARY'

    dataset = TUDataset(root='../../dataset', name=dataset_name)

    config = edict({
        "in_dim": dataset.num_features,
        "hid_dim": 64,
        "out_dim": dataset.num_classes,
        "lr": 0.01,
        "epochs": 200,
        "weight_decay": 5e-4,
        "task_name": "GraphClassification",
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # "device": "cpu",
        "use_parallel": False,
        "layers": 2,
        "dropout": 0.5,
        "model": 'Auto-GNN',
        "gpu": 0,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "batch_size": 32,
        "optimizer": "Adam",
        "dataset_name": dataset_name,
        "seed": 10,
    })
    # set_random_seed(config.seed)

    import torch.nn.functional as F
    import torch.nn as nn
    from torch_geometric.nn import GATConv, GCN2Conv, GATv2Conv
    from llm4gnas.search_space.autogel_space import global_add_pool, global_mean_pool, global_max_pool, \
        FeedForwardNetwork

    device = config.device


    class GCN(nn.Module):

        def __init__(self, in_channels, hidden_channels, out_channels):
            super(GCN, self).__init__()

            self.lin1 = nn.Linear(in_channels, hidden_channels)
            # self.conv1 = GCN2Conv(in_channels, hidden_channels)
            # self.conv2 = GCN2Conv(hidden_channels, out_channels)

            self.conv1 = GATv2Conv(in_channels, hidden_channels)
            self.conv2 = GATv2Conv(hidden_channels, hidden_channels)
            self.feed_forward = FeedForwardNetwork(hidden_channels, out_channels)

        def forward(self, batch):
            x = batch.x
            # x = self.preprocess(x)

            edge_index = batch.edge_index
            # x_0 = self.lin1(x)
            if batch.num_node_features == 0:
                x = torch.ones((batch.num_nodes, 1), device=device)
            x = F.dropout(x, p=0.6, training=self.training)

            # x = F.relu(self.conv1(x=x, edge_index=edge_index, x_0=x_0))
            # x = self.conv2(x=x, edge_index=edge_index, x_0=x_0)

            x = F.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)

            x = self.global_pool_trans(x, batch.batch, torch.tensor([[1., 0., 0.]], device=device))
            x = self.feed_forward(x)

            return x

        def global_pool_trans(self, x, batch, z_hard):
            y = []
            for pool in {'global_add_pool': [1.0, 0.0, 0.0], 'global_mean_pool': [0.0, 1.0, 0.0],
                         'global_max_pool': [0.0, 0.0, 1.0]}:
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

        def loss(self, data: Data, out):
            loss = F.cross_entropy(out, data.y)
            return loss

        def metric(self, data: Tuple, gnn: GNNBase):
            train_loader, val_loader, test_loader = data
            metric = {"train acc": .0, "val acc": .0, "test acc": .0}
            loader_dict = {
                "train": train_loader,
                "val": val_loader,
                "test": test_loader
            }
            for loader_name, loader in loader_dict.items():
                correct = 0
                total = 0
                for batch in loader:
                    batch = batch.to(device)
                    out = gnn(batch)
                    pred = out.argmax(dim=1)
                    # print(f"pred: {pred}")
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
                # print(f"correct: {correct}, total: {total}")
                metric[f"{loader_name} acc"] = correct / total if total > 0 else 0.0
            return metric


    # desc = ['gcn', 'gat', 'gat', 'gat']
    # desc = "{layer1:{ agg:max, combine:sum, act:relu, layer_connect:skip_cat}; layer2:{ agg:mean, combine:concat, act:prelu, layer_connect:stack}; layer_agg:concat;}"

    # Trainer = model_factory["graph_trainer"](config)
    search_space = model_factory["autogel_space"](config)
    Trainer = model_factory["graph_trainer"](config)
    # search_space = model_factory["gcn_only_space"](config)
    # model = search_space.to_gnn(desc=desc)
    model = GCN(config.in_dim, config.hid_dim, config.out_dim)
    # set_random_seed(i + 326)
    for _ in range(12):
        gnn = Trainer.fit(dataset, model)
        metric = Trainer.evaluate(dataset, gnn)
        print(metric)
    # pred = Trainer.predict(dataset)
    # print(pred)
