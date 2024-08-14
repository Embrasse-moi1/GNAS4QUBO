import pathlib
from torch_geometric.datasets import Planetoid
from for_other_dataset_exp.llm4gnas.args import *
from for_other_dataset_exp.llm4gnas.autosolver import *
from for_other_dataset_exp.llm4gnas.args import *
from for_other_dataset_exp.llm4gnas.utils.utils import set_random_seed


if __name__ == '__main__':
    home_dir = pathlib.Path.home()
    dataset_name = "citeseer"
    dataset = Planetoid(f"{home_dir}/GNAS4CO/dataset", dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = edict({
        "in_dim": 3703,
        "hid_dim": 64,
        "out_dim": 6,
        "lr": 0.01,
        "epochs": 200,
        "weight_decay": 5e-4,
        "device": device,
        "parallel": False,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "task_name": "NodeClassification",
        "layers": 2,
        "dropout": 0.5,
        "model": 'Auto-GNN',  # ???
        "gpu": 0,
        "batch_size": 64,
        "optimizer": "Adam",
        "dataset_name": dataset_name,
        "seed": 18,
    })

    desc = "{layer1:{ agg:mean, combine:sum, act:prelu, layer_connect:skip_sum}; layer2:{ agg:sum, combine:concat, act:relu, layer_connect:stack}; layer_agg:max_pooling;}"
    Trainer = model_factory["normal_trainer"](config)
    search_space = model_factory["autogel_space"](config)

    set_random_seed(config.seed)
    model = search_space.to_gnn(desc=desc)
    gnn = Trainer.fit(dataset, model)
    metric = Trainer.evaluate(dataset, gnn)
    print(metric)
