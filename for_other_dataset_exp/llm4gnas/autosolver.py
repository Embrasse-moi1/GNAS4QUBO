from torch_geometric.data import Data, Dataset
from for_other_dataset_exp.llm4gnas.search_space import *
from for_other_dataset_exp.llm4gnas.trainer import *
from for_other_dataset_exp.llm4gnas.nas_method import *
from for_other_dataset_exp.llm4gnas.register import model_factory
from easydict import EasyDict as edict


class AutoSolver(object):

    def __init__(
            self,
            search_space: str,  # Search space name, defining the scope of architecture search
            nas_method: str,  # NAS method name, determining how architecture search is conducted
            training_method: str,  # Training method name, defining the strategy for model training
            config: dict,  # Hyperparameter configuration, including device selection and other settings
    ):
        # Validate if the search space, NAS method, and training method exist in the predefined model_factory
        assert search_space in model_factory, f"Search space:{search_space} not exists"
        assert nas_method in model_factory, f"Nas method:{nas_method} not exists"
        assert training_method in model_factory, f"Training method:{training_method} not exists"

        # Initialize configuration; if no device specified, automatically choose CUDA or CPU
        # self.config = edict(config)
        self.config = config
        if not hasattr(self.config, "device"):
            self.config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_gnn = None  # Initialize the best GNN model as None

        # Initialize search space object, trainer, and NAS method based on the provided search space and training method names
        self.search_space = model_factory[search_space](self.config)
        self.trainer = model_factory[training_method](self.config)
        self.nas_method = model_factory[nas_method](self.search_space, self.trainer, self.config)

    def fit(self, dataset: Union[Data, Dataset, None]) -> GNNBase:
        """
        Trains a GNN model using the specified NAS method to fit the given data.

        Parameters:
        - data: Data - A data object providing the training data.

        Returns:
        - best_gnn: GNNBase - The trained instance of the best GNN model found.
        """
        best_gnn = self.nas_method.fit(dataset)  # Train GNN using NAS method to find optimal structure
        self.best_gnn = best_gnn  # Store the found best GNN model in the instance
        return best_gnn

    def evaluate(self, dataset: Union[Data, Dataset, None], gnn: Union[GNNBase, None] = None) -> dict:
        """
        Evaluate given data.

        Parameters:
        - data: Data, The dataset to be evaluated.
        - gnn: Union[GNNBase, None], Optional, An instance of GNN model. If None, the default model will be used for evaluation.

        Returns:
        - dict, A dictionary containing evaluation metrics.
        """
        # Invoke the `evaluate` method of the trainer to perform model evaluation
        return self.trainer.evaluate(dataset, gnn)

    def predict(self, dataset: Union[Data, Dataset, None], gnn: Union[GNNBase, None] = None) -> Tensor:
        """
        Perform prediction on the given data using a specified GNN model or the best GNN model.

        Parameters:
        - data: Data, the data to be predicted.
        - gnn: Union[GNNBase, None], the GNN model to use for prediction. If None, the best GNN model from the trainer will be used.

        Returns:
        - Tensor, a tensor containing the prediction results.
        """
        gnn = self.best_gnn if gnn is None else gnn  # Use the best GNN model if no specific GNN is provided
        return self.trainer.predict(dataset, gnn)  # Perform prediction using the trainer's predict method

    def get_model(self) -> GNNBase:
        return self.best_gnn
