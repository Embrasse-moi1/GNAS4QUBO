from for_other_dataset_exp.llm4gnas.trainer.trainer_base import *
from for_other_dataset_exp.llm4gnas.register import model_factory


class NASBase(object):
    def __init__(self, search_space: SearchSpaceBase, trainer: TrainerBase, config: dict, **kwargs):
        self.required_prompts = []
        self.config = config
        self.search_space = search_space
        self.trainer = trainer
        self.llm = None
        self.best_model = None

    def fit(self, data) -> GNNBase:
        raise NotImplementedError

    def reset(self) -> None:
        self.llm = None
        self.best_model = None


class DummyNAS(NASBase):
    def fit(self, data) -> GNNBase:
        gnn = self.search_space.to_gnn(["gcn", "gcn"])
        gnn = self.trainer.fit(data, gnn, self.config)
        self.best_model = gnn
        return gnn


model_factory["dummy_nas"] = DummyNAS
