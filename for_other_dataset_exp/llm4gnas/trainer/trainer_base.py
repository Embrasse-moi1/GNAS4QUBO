from typing import Any

from torch_geometric.data import Dataset
from tqdm import tqdm
import ray
from for_other_dataset_exp.llm4gnas.search_space import *
from for_other_dataset_exp.llm4gnas.utils.data import get_optimizer, get_loader


class TrainerBase(object):
    def __init__(self, config: dict, **kwargs):
        self.config = config
        self.gnn = None
        self.optimizer = None
        self.device = config.device

    def fit(self, dataset: Dataset, gnn: GNNBase, config: dict = None) -> GNNBase:
        pass

    def evaluate(self, dataset: Dataset, gnn: Union[GNNBase, None] = None) -> dict:
        # return metrics
        raise NotImplementedError

    def predict(self, dataset: Dataset, gnn: Union[GNNBase, None] = None):
        raise NotImplementedError

    def get_result(self, object_ref: Any) -> Any:
        if self.config.parallel:
            return ray.get(object_ref)
        else:
            return object_ref


def ray_remote_decorator(func):
    @ray.remote(num_gpus=torch.cuda.device_count())
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


class GlobalBatchTrainer(TrainerBase):
    def __init__(self, config: dict, **kwargs):

        super().__init__(config, **kwargs)
        if self.config.task_name == "GraphClassification":
            raise ValueError("Graph classification tasks require the use of graph_trainer")
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def fit(self, dataset: Dataset, gnn: GNNBase, config: dict = None) -> GNNBase:
        config = self.config if config is None else config
        if config is None:
            raise ValueError("config is None.")

        self.optimizer = get_optimizer(gnn, config)

        # optimizer = torch.optim.Adam(gnn.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        data = dataset[0]
        self.gnn = gnn.to(self.device)
        data = data.to(self.device)
        if config.parallel:
            use_ray = ray_remote_decorator(self.gpu_fit)
            self.gnn = use_ray.remote(data)
        else:
            self.gnn = self.gpu_fit(data)
        return self.gnn

    def evaluate(self, dataset: Dataset, gnn: Union[GNNBase, None] = None) -> dict:
        data = dataset[0]
        data= data.to(self.device)
        gnn = self.gnn if gnn is None else gnn
        if gnn is None and gnn is None:
            raise RuntimeError("GNN is not Given.")
        gnn.eval()
        return gnn.metric(data, gnn(data))

    def predict(self, dataset: Dataset, gnn: Union[GNNBase, None] = None):
        data = dataset[0]
        data = data.to(self.device)
        gnn = self.gnn if gnn is None else gnn
        if gnn is None and self.gnn is None:
            raise RuntimeError("GNN is not Given.")
        gnn.eval()
        return gnn(data)

    def gpu_fit(self, data: Data) -> GNNBase:
        gnn = self.gnn
        for epoch in tqdm(range(1, self.config.epochs + 1)):
            gnn.train()
            self.optimizer.zero_grad()
            loss = gnn.loss(data=data, out=gnn(data))
            loss.backward(retain_graph=True)
            self.optimizer.step()
        return gnn


class NormalTrainer(TrainerBase):
    def fit(self, dataset: Dataset, gnn: GNNBase, config: dict = None) -> GNNBase:
        data = dataset
        self.gnn = gnn
        config = self.config if config is None else config
        gnn = gnn.to(self.device)
        optimizer = torch.optim.Adam(gnn.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        for epoch in tqdm(range(1, config.epochs + 1)):
            gnn.train()
            optimizer.zero_grad()
            loss = gnn.loss(data=data, out=gnn(data))
            loss.backward(retain_graph=True)
            optimizer.step()
        return gnn

    def evaluate(self, dataset: Dataset, gnn: Union[GNNBase, None] = None) -> dict:
        data = dataset
        gnn = self.gnn if gnn is None else gnn
        if gnn is None and gnn is None:
            raise RuntimeError("GNN is not Given.")
        gnn.eval()
        return gnn.metric(data, gnn(data))

    def predict(self, dataset: Dataset, gnn: Union[GNNBase, None] = None) -> Tensor:
        data = dataset[0]
        gnn = self.gnn if gnn is None else gnn
        if gnn is None and self.gnn is None:
            raise RuntimeError("GNN is not Given.")
        gnn.eval()
        return gnn(data)


model_factory["global_batch"] = GlobalBatchTrainer
model_factory["normal_trainer"] = NormalTrainer

if __name__ == "__main__":
    from torch_geometric.datasets import Planetoid
    from easydict import EasyDict as edict

    dataset_name = "pubmed"
    dataset = Planetoid("../../dataset", dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data = dataset[0].to(device)
    config = edict({
        "in_dim": dataset.num_features,
        "hid_dim": 64,
        "out_dim": dataset.num_classes,
        "lr": 0.01,
        "epochs": 200,
        "weight_decay": 5e-4,
        "device": device,
        "parallel": False,
        "dataset_name": "cora",
        "task_name": "NodeClassification",
        "layers": 2,
        "dropout": 0.5,
        "model": 'Auto-GNN',  # ???
        "gpu": 0,
        "batch_size": 64,
        "optimizer": "Adam",
    })

    # desc = ['gat', 'gat', 'gat', 'gat']
    desc = "{layer1:{ agg:max, combine:sum, act:relu, layer_connect:skip_cat}; layer2:{ agg:mean, combine:concat, act:prelu, layer_connect:stack}; layer_agg:concat;}"
    Trainer = model_factory["global_batch"](config)
    search_space = model_factory["autogel_space"](config)
    # search_space = model_factory["autogel_space"](config)
    model = search_space.to_gnn(desc=desc)
    gnn = Trainer.fit(dataset, model)
    gnn = Trainer.get_result(gnn)
    metric = Trainer.evaluate(dataset, gnn)
    print(metric)
