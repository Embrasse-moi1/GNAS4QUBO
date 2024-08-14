from for_other_dataset_exp.llm4gnas.autosolver import AutoSolver
from for_other_dataset_exp.llm4gnas.args import get_args
from for_other_dataset_exp.llm4gnas.utils.data import load_data
from for_other_dataset_exp.llm4gnas.utils.utils import set_random_seed


if __name__ == '__main__':
    config = get_args()
    set_random_seed(config.seed)
    dataset = load_data(config)

    solver = AutoSolver(
        search_space="autogel_space",
        nas_method="gpt4gnas",
        training_method="link_trainer",
        config=config
    )
    best_gnn = solver.fit(dataset)
    print(best_gnn.desc)

    metric = solver.evaluate(dataset, best_gnn)
    print(metric)
