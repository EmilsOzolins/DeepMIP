from typing import Any

import ecole as ec
import numpy as np
import torch

import config as config
import hyperparams as params
from data.item_placement import ItemPlacementDataset
from data.load_balancing import LoadBalancingDataset
from model.mip_network import MIPNetwork
from utils.data_utils import InputDataHolder


class ObservationFunction(ec.observation.NodeBipartite):  # This allows customizing information received by the solver

    def __init__(self, problem):
        super(ObservationFunction, self).__init__()
        # called once for each problem benchmark
        self.problem = problem  # to devise problem-specific observations
        self.previous_file_name = None

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        pass

    def before_reset(self, model):
        pass

    def extract(self, model, done):
        m = model.as_pyscipopt()
        variables = m.getVars(transformed=True)

        # Extract the upper and lower bounds of all variables
        lbs = np.asarray([v.getLbLocal() for v in variables])
        ubs = np.asarray([v.getUbLocal() for v in variables])

        # extract features of variables and constraints
        obs = super().extract(model, done)
        column_features = obs.column_features  # Variables features
        row_features = obs.row_features  # Constraints features
        edge_features = obs.edge_features  # Variables-Constraints graph

        return {"lbs": lbs,
                "ubs": ubs,
                "column_features": column_features,
                "row_features": row_features,
                "edge_features": edge_features}


class MIPObservationHolder(InputDataHolder):

    def __init__(self, observations: dict, device: torch.device) -> None:
        super().__init__()
        self._observations = observations
        self._device = device

    @property
    def vars_const_graph(self) -> torch.sparse.Tensor:
        graph = self._observations["edge_features"]  # TODO: It could be that we need to transpose this matrix
        indices = np.int64(graph.indices)
        torch_graph = torch.sparse_coo_tensor(indices, graph.values,
                                              graph.shape, dtype=torch.float32,
                                              device=self._device).t()
        return torch_graph.coalesce()

    @property
    def binary_vars_const_graph(self) -> torch.sparse.Tensor:
        graph = self._observations["edge_features"]
        torch_graph = torch.sparse_coo_tensor(np.int64(graph.indices),
                                              torch.ones_like(graph.values), graph.shape,
                                              dtype=torch.float32, device=self._device).t()
        return torch_graph.coalesce()

    @property
    def const_values(self) -> torch.Tensor:
        bias = self._observations["row_features"][:, 0]
        return torch.as_tensor(bias, device=self._device, dtype=torch.float32)

    @property
    def vars_obj_graph(self) -> torch.sparse.Tensor:
        graph = self._observations["edge_features"]
        objective_mul = self._observations["column_features"][:, 0]
        _, vars_count = graph.shape
        indices_var = [x for x in range(vars_count)]
        indices_obj = [0] * vars_count

        torch_graph = torch.sparse_coo_tensor((indices_var, indices_obj),
                                              objective_mul, [vars_count, 1],
                                              dtype=torch.float32, device=self._device)
        return torch_graph.coalesce()

    @property
    def const_inst_graph(self) -> torch.sparse.Tensor:
        graph = self._observations["edge_features"]
        const_count, _ = graph.shape
        indices_var = [x for x in range(const_count)]
        indices_inst = [0] * const_count
        torch_graph = torch.sparse_coo_tensor((indices_var, indices_inst),
                                              torch.ones([const_count]), [const_count, 1],
                                              dtype=torch.float32, device=self._device)

        return torch_graph.coalesce()

    @property
    def vars_inst_graph(self) -> torch.sparse.Tensor:
        graph = self._observations["edge_features"]
        _, vars_count = graph.shape
        indices_var = [x for x in range(vars_count)]
        indices_inst = [0] * vars_count
        torch_graph = torch.sparse_coo_tensor((indices_var, indices_inst), torch.ones([vars_count]),
                                              size=[vars_count, 1], dtype=torch.float32, device=self._device)
        return torch_graph.coalesce()

    @property
    def optimal_solution(self) -> torch.Tensor:
        return torch.tensor([float("nan")], device=self._device, dtype=torch.float32)

    @property
    def integer_mask(self) -> torch.Tensor:
        binary_mask = self._observations["column_features"][:, 1]
        integer_mask = self._observations["column_features"][:, 2]
        mask = np.logical_or(binary_mask, integer_mask)
        return torch.as_tensor(mask, device=self._device, dtype=torch.float32)

    @property
    def objective_multipliers(self) -> torch.Tensor:
        mul = self._observations["column_features"][:, 0]
        return torch.as_tensor(mul, device=self._device, dtype=torch.float32)

    def get_data(self, *keys: str) -> Any:
        pass


class Policy():
    # TODO: Run on the train and validate datasets

    def __init__(self, problem):
        # called once for each problem benchmark
        self.rng = np.random.RandomState()
        self.problem = problem  # to devise problem-specific policies

        self.device = torch.device(config.device)
        self.network = MIPNetwork(
            output_bits=params.output_bits,
            feature_maps=params.feature_maps,
            pass_steps=params.recurrent_steps,
            summary=None
        )
        run_name = "20210927-155836"
        checkpoint = torch.load(f"/host-dir/mip_models/{run_name}/model.pth")

        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.network.eval()
        self.network.to(self.device)

        self.dataset = ItemPlacementDataset(
            "/host-dir/") if self.problem == "item_placement" else LoadBalancingDataset("/host-dir/")

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        self.rng = np.random.RandomState(seed)

    def __call__(self, action_set, observation):
        obs_holder = MIPObservationHolder(observation, self.device)

        with torch.no_grad():
            outputs, logits = self.network.forward(obs_holder, self.device)
        output = self.dataset.decode_model_outputs(outputs[-1], obs_holder)

        return action_set, output[np.int64(action_set)].cpu().numpy()
