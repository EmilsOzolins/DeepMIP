import glob
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from data.datasets_base import MIPDataset
from data.mip_instance import MIPInstance
from metrics.general_metrics import Metrics
from metrics.mip_metrics import MIPMetrics
from utils.data import MIPBatchHolder


class ItemPlacementDataset(MIPDataset, Dataset):

    def __init__(self, data_folder, augment: bool = False) -> None:
        self._instances = glob.glob(data_folder + "/*.npz")
        self._should_augment = augment

    @property
    def required_output_bits(self):
        return 1

    @property
    def test_metrics(self) -> List[Metrics]:
        return [MIPMetrics()]

    @property
    def train_metrics(self) -> List[Metrics]:
        return []

    def decode_model_outputs(self, model_output, batch_holder: MIPBatchHolder):
        output = torch.squeeze(model_output)
        mask = batch_holder.integer_mask
        inv_mask = 1 - mask
        return torch.round(output * mask) + output * inv_mask

    def __getitem__(self, index: int) -> Dict:
        file_name = self._instances[index]
        data = np.load(file_name)

        variables_features = data["variables_features"]
        constraints_features = data["constraints_features"]
        edges = data["edges"].astype(np.int64)
        edge_values = data["edge_values"]
        graph_shape = data["graph_shape"]
        # variables_lbs = data["variables_lbs"]
        # variables_ubs = data["variables_ubs"]

        integer_variables = variables_features[:, 2]
        binary_variables = variables_features[:, 1]
        obj_multipliers = variables_features[:, 0]
        bias_values = constraints_features[:, 0]

        const_count, var_count = graph_shape

        # TODO: Speed up this and do preprocessing only once
        constraints = [([], []) for _ in range(const_count)]

        for (c_id, var_id), val in zip(np.transpose(edges), edge_values):
            constraints[c_id][0].append(var_id)
            constraints[c_id][1].append(val)

        mip = MIPInstance(var_count)

        for const, bias in zip(constraints, bias_values):
            mip.less_or_equal(const[0], const[1], bias)

        # for vid, lb in enumerate(variables_lbs):
        #     mip.greater_or_equal([vid],[1], lb)
        #
        # for vid, ub in enumerate(variables_ubs):
        #     mip.less_or_equal([vid],[1], ub)

        mip.minimize_objective([i for i in range(var_count)], obj_multipliers)
        int_constraints = [i for i, (bv, iv) in enumerate(zip(integer_variables, binary_variables)) if bv or iv]
        mip.integer_constraint(int_constraints)

        return {"mip": mip.augment() if self._should_augment else mip,
                "optimal_solution": torch.as_tensor([float('nan')], dtype=torch.float32)}

    def __len__(self) -> int:
        return len(self._instances)
