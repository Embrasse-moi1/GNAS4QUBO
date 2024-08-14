import torch
from torch import Tensor
from tqdm import tqdm
from typing import Union
from torch_geometric.data import Data, Dataset
from for_other_dataset_exp.llm4gnas.register import model_factory
from for_other_dataset_exp.llm4gnas.utils.data import get_loader, get_optimizer
from for_other_dataset_exp.llm4gnas.trainer.trainer_base import TrainerBase
from for_other_dataset_exp.llm4gnas.search_space import GNNBase
from sklearn.metrics import roc_auc_score
criterion = torch.nn.functional.cross_entropy
class LinkPredictTrainer(TrainerBase):
    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)
        if config.task_name != "LinkPredict":
            raise ValueError(f"Task name should be LinkPredict, but got {config.task_name}")
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def fit(self, dataset: Dataset, gnn: GNNBase, config: dict = None) -> GNNBase:
        config = self.config if config is None else config
        if config is None:
            raise ValueError("config is None.")

        if self.train_loader is None or self.val_loader is None or self.test_loader is None:
            self.train_loader, self.val_loader, self.test_loader = dataset

        self.optimizer = get_optimizer(gnn, config)
        self.gnn = gnn.to(self.device)
        self.gnn.train()
        for epoch in tqdm(range(1, config.epochs + 1)):
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                label = batch.y
                out = self.gnn(batch)
                loss = criterion(out, label, reduction='mean')
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), max_norm=1)
                self.optimizer.step()
        return self.gnn

    def evaluate(self, dataset: Dataset, gnn: Union[GNNBase, None] = None, return_predictions=False) -> dict:
        gnn = self.gnn if gnn is None else gnn.to(self.device)
        if gnn is None and gnn is None:
            raise RuntimeError("GNN is not Given.")
        gnn.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gnn.to(device)
        predictions = []
        labels = []
        with torch.no_grad():
            for batch in dataset:
                batch = batch.to(device)
                labels.append(batch.y)
                prediction = gnn(batch)
                predictions.append(prediction)
            predictions = torch.cat(predictions, dim=0)
            labels = torch.cat(labels, dim=0)
        if not return_predictions:
            loss, acc, auc = compute_metric(predictions, labels)
            print(loss, acc, auc)
            loss, acc, auc = float(loss), float(acc), float(auc)
            return {"train acc": loss, "val acc": acc, "test acc": auc}
        else:
            return predictions
    def predict(self, data: Dataset, gnn: Union[GNNBase, None] = None) -> Tensor:
        data = data[0].to(self.device)
        # pred = []
        gnn = self.gnn if gnn is None else gnn
        if gnn is None and self.gnn is None:
            raise RuntimeError("GNN is not Given.")
        gnn.eval()
        return gnn(data)

def compute_metric(predictions, labels):
    with torch.no_grad():
        # compute loss:
        loss = criterion(predictions, labels, reduction='mean').item()
        # compute acc:
        correct_predictions = (torch.argmax(predictions, dim=1) == labels)
        acc = correct_predictions.sum().cpu().item()/labels.shape[0]
        # compute auc:
        predictions = torch.nn.functional.softmax(predictions, dim=-1)
        multi_class = 'ovr'
        if predictions.size(1) == 2:
            predictions = predictions[:, 1]
            multi_class = 'raise'
        auc = roc_auc_score(labels.cpu().numpy(), predictions.cpu().numpy(), multi_class=multi_class)
    return loss, acc, auc

model_factory["link_trainer"] = LinkPredictTrainer