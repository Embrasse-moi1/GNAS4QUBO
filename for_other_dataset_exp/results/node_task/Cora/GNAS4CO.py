import pathlib
from torch_geometric.datasets import Planetoid
from for_other_dataset_exp.llm4gnas.search_space import *
from for_other_dataset_exp.llm4gnas.args import *
from for_other_dataset_exp.llm4gnas.autosolver import *
from for_other_dataset_exp.llm4gnas.utils.utils import set_random_seed
from for_other_dataset_exp.llm4gnas.register import model_factory

if __name__ == '__main__':
    home_dir = pathlib.Path.home()
    dataset_name = "cora"
    dataset = Planetoid(f"{home_dir}/GNAS4CO/dataset", dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = edict({
        "in_dim": 1433,
        "hid_dim": 64,
        "out_dim": 7,
        "lr": 0.01,
        "epochs": 200,
        "weight_decay": 5e-4,
        "device": device,
        "parallel": False,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "batch_size": 64,
        "task_name": "NodeClassification",
        "layers": 2,
        "dropout": 0.5,
        "model": 'Auto-GNN',  # ???
        "gpu": 0,
        "optimizer": "Adam",
        "dataset_name": dataset_name,
        "seed": 18,
    })

    dataset.num_nodes = len(dataset.y)
    num_nodes = dataset.num_nodes
    node_id = np.arange(num_nodes)
    np.random.shuffle(node_id)
    dataset.train_id = np.sort(node_id[:int(num_nodes * 0.6)])
    dataset.val_id = np.sort(
        node_id[int(num_nodes * 0.6):int(num_nodes * 0.8)])
    dataset.test_id = np.sort(node_id[int(num_nodes * 0.8):])

    dataset.train_mask = torch.tensor(
        [x in dataset.train_id for x in range(num_nodes)])
    dataset.val_mask = torch.tensor(
        [x in dataset.val_id for x in range(num_nodes)])
    dataset.test_mask = torch.tensor(
        [x in dataset.test_id for x in range(num_nodes)])

    dataset = dataset.to(device)

    desc = "{layer1:{ agg:mean, combine:sum, act:relu, layer_connect:skip_cat}; layer2:{ agg:sum, combine:concat, act:prelu, layer_connect:stack}; layer_agg:none;}"
    Trainer = model_factory["normal_trainer"](config)
    search_space = model_factory["autogel_space"](config)
    model = search_space.to_gnn(desc=desc)
    gnn = Trainer.fit(dataset, model)
    set_random_seed(config.seed)
    # gnn = Trainer.get_result(gnn)
    metric = Trainer.evaluate(dataset, gnn)
    print(metric)

