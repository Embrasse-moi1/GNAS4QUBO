from torch_geometric.datasets import TUDataset
from for_other_dataset_exp.llm4gnas.search_space import *
from for_other_dataset_exp.llm4gnas.trainer import *
from for_other_dataset_exp.llm4gnas.nas_method import *
from for_other_dataset_exp.llm4gnas.utils.utils import set_random_seed
from for_other_dataset_exp.llm4gnas.register import model_factory
from for_other_dataset_exp.llm4gnas.utils.data import *
import torch
from easydict import EasyDict as edict
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = edict({
        "graph_type": "homo",
        "in_dim": 0,
        "hid_dim": 64,
        "out_dim": 0,
        "input": "",
        "lr": 0.01,
        "epochs": 200,
        "weight_decay": 5e-4,
        "task_name": "GraphClassification",
        "device": device,
        # "device": "cpu",
        "parallel": False,
        "layers": 2,
        "dropout": 0.5,
        "model": 'Auto-GNN',  # ???
        "gpu": 0,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "batch_size": 128,
        "optimizer": "Adam",
        "dataset_name": "PROTEINS",
        "seed": 25,
    })
    set_random_seed(config.seed)
    dataset = load_data(config)
    desc = "{layer1:{ agg:sum, combine:concat, act:prelu, layer_connect:skip_sum}; layer2:{ agg:mean, combine:sum, act:relu, layer_connect:skip_cat}; layer_agg:max_pooling;}"
    Trainer = model_factory["graph_trainer"](config)
    search_space = model_factory["autogel_space"](config)
    model = search_space.to_gnn(desc=desc)
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    for epoch in range(20):
        gnn = Trainer.fit(dataset, model)
        metric = Trainer.evaluate(dataset, gnn)
        train_acc_list.append(metric["train acc"])
        val_acc_list.append(metric["val acc"])
        test_acc_list.append(metric["test acc"])

    avg_train_acc = np.mean(train_acc_list)
    avg_val_acc = np.mean(val_acc_list)
    avg_test_acc = np.mean(test_acc_list)
    # std
    std_train_acc = np.std(train_acc_list)
    std_val_acc = np.std(val_acc_list)
    std_test_acc = np.std(test_acc_list)
    print("train acc: %.4f±%.4f, val acc %.4f±%.4f, test acc %.4f±%.4f" % (
        avg_train_acc,
        std_train_acc,
        avg_val_acc,
        std_val_acc,
        avg_test_acc,
        std_test_acc
    ))
