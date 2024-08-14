import os.path
import yaml
import argparse
import torch


def get_args(description="GNAS4CO", config_file=""):
    """
    Use settings in config file as defaults
    :param description:
    :param config_file:
    :return:
    """
    parser = argparse.ArgumentParser(description=description)
    # get config first
    parser.add_argument("--config", type=str, default=config_file)
    given_configs, remaining = parser.parse_known_args()

    # read default setting from config file
    register_args(parser)
    if given_configs.config:
        if os.path.exists(given_configs.config):
            with open(given_configs.config, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)
        else:
            raise RuntimeError(f"Config file {given_configs.config} not exists")

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.

    args = parser.parse_args(remaining)
    if "cuda" in args.device:
        args.device = args.device if torch.cuda.is_available() else "cpu"

    if args.dataset_name in ["acm", "dblp", "imdb", "amazon", "yelp", "movielens"]:
        args = parser.parse_args(remaining)
        args.graph_type = "hetero"
        if args.task_name == "LinkPredict":
            args.metrics = "auc"

    else:
        if args.task_name == "NodeClassification":
            register_args_autogel(parser)
            args = parser.parse_args(remaining)
        else:
            lp = True
            register_args_autogel(parser, lp=lp)
            args = parser.parse_args(remaining)
    return args


def register_args(parser):
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--graph_type", type=str, choices=["homo", "hetero"], default="homo")

    # LLM related
    parser.add_argument("--llm", type=str, default="ChatGPTProxy")
    parser.add_argument("--llm_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--api_key", type=str, default="xxx")
    parser.add_argument("--use_gpt_4_tape", type=str, default=False)

    # NAS related
    parser.add_argument("--dataset_name", type=str, default="dblp")
    parser.add_argument("--nas_iterations", type=int, default=1)

    # GNN related
    parser.add_argument("--in_dim", type=int, default=0)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--out_dim", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    # data infos
    parser.add_argument("--input", type=str, default="", help='Path of custom dataset')
    parser.add_argument("--data_transform", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--task_name", type=str, default="NodeClassification")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer to use')

    # llm hpo related
    parser.add_argument("--hpo_iterations", type=int, default=3)

    # parallel related
    parser.add_argument("--parallel", type=bool, default=False)
    parser.add_argument('--debug', action="store_true", default=False)

    return parser


# autogel args
def register_args_autogel(parser, lp=None):
    if lp is None:
        # parser.add_argument('--train_ratio', type=float, default=0.6, help='ratio of the train against whole')
        # parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio of the val against whole')
        # parser.add_argument('--test_ratio', type=float, default=0.2, help='ratio of the test against whole')
        parser.add_argument('--model', type=str, default='Auto-GNN', help='model to use')
        parser.add_argument('--metric', type=str, default='acc', help='metric for evaluating performance',
                            choices=['acc', 'auc'])
        parser.add_argument('--gpu', type=int, default=0, help='gpu id')
        parser.add_argument('--directed', type=bool, default=False,
                            help='(Currently unavailable) whether to treat the graph as directed')
        # parser.add_argument('--parallel', default=False, action='store_true',
        #                     help='(Currently unavailable) whether to use multi cpu cores to prepare data')

        # parser.add_argument('--task', type=str, default='NodeClassification', help='type of task',choices=['NodeClassification', "GraphClassification", "LinkPredict"])
        parser.add_argument('--seed', type=int, default=10, help='seed to initialize all the random modules')
        parser.add_argument('--clip', type=float, default=0.1, help='gradient clipping')
        # general model and training setting
        # parser.add_argument('--dataset', type=str, default='Cora',help='dataset name')  # choices=['PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'BZR', 'COX2', 'DD', 'ENZYMES', 'NCI1']

        # parser.add_argument('--epoch', type=int, default=20, help='number of epochs to search')
        parser.add_argument('--retrain_epoch', type=int, default=200, help='number of epochs to retrain')

        parser.add_argument('--layers', type=int, default=2, help='largest number of layers')
        # parser.add_argument('--hidden_features', type=int, default=128, help='hidden dimension')
        parser.add_argument('--bs', type=int, default=128, help='minibatch size')
        # parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
        parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight')
        parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
        # parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
        # logging & debug
        parser.add_argument('--log_dir', type=str, default='./log/', help='log directory')

        parser.add_argument('--summary_file', type=str, default='result_summary.log',
                            help='brief summary of training result')
    else:

        parser.add_argument('--train_ratio', type=float, default=0.6, help='ratio of the train against whole')
        parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio of the val against whole')
        parser.add_argument('--test_ratio', type=float, default=0.2, help='ratio of the test against whole')
        # general model and training setting
        parser.add_argument('--gpu', type=int, default=0, help='gpu id')

        # parser.add_argument('--dataset', type=str, default='celegans_small', help='dataset name')  # currently relying on dataset to determine task
        # parser.add_argument('--task', type=str, default='LinkPredict', help='type of task', choices=['NodeClassification', "GraphClassification", "LinkPredict"])
        # parser.add_argument('--test_ratio', type=float, default=0.1, help='ratio of the test against whole')
        # parser.add_argument('--model', type=str, default='Auto-GNN', help='model to use', choices=['DE-GNN', 'GIN', 'GCN', 'GraphSAGE', 'GAT'])
        parser.add_argument('--model', type=str, default='Auto-GNN', help='model to use')
        parser.add_argument('--layers', type=int, default=2, help='largest number of layers')
        # parser.add_argument('--hidden_features', type=int, default=100, help='hidden dimension')
        parser.add_argument('--metric', type=str, default='auc', help='metric for evaluating performance',
                            choices=['acc', 'auc'])
        parser.add_argument('--seed', type=int, default=3, help='seed to initialize all the random modules')

        # parser.add_argument('--adj_norm', type=str, default='asym', help='how to normalize adj', choices=['asym', 'sym', 'None'])
        parser.add_argument('--data_usage', type=float, default=1.0, help='use partial dataset')
        parser.add_argument('--directed', type=bool, default=False,
                            help='(Currently unavailable) whether to treat the graph as directed')
        # parser.add_argument('--parallel', default=False, action='store_true',
        #                     help='(Currently unavailable) whether to use multi cpu cores to prepare data')

        # features and positional encoding
        parser.add_argument('--prop_depth', type=int, default=1,
                            help='propagation depth (number of hops) for one layer')
        parser.add_argument('--use_degree', type=bool, default=True,
                            help='whether to use node degree as the initial feature')
        parser.add_argument('--use_attributes', type=bool, default=False,
                            help='whether to use node attributes as the initial feature')
        parser.add_argument('--feature', type=str, default='sp',
                            help='distance encoding category: shortest path or random walk (landing probabilities)')  # sp (shortest path) or rw (random walk)
        parser.add_argument('--rw_depth', type=int, default=3, help='random walk steps')  # for random walk feature
        parser.add_argument('--max_sp', type=int, default=50,
                            help='maximum distance to be encoded for shortest path feature')

        # model training
        # parser.add_argument('--epoch', type=int, default=300, help='number of epochs to search')
        parser.add_argument('--retrain_epoch', type=int, default=300, help='number of epochs to retrain')
        parser.add_argument('--bs', type=int, default=64, help='minibatch size')
        # parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
        # parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
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
        parser.add_argument('--log_dir', type=str, default='./log/',
                            help='log directory')  # sp (shortest path) or rw (random walk)

        parser.add_argument('--summary_file', type=str, default='result_summary.log',
                            help='brief summary of training result')  # sp (shortest path) or rw (random walk)
