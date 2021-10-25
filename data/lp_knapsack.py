import random
from typing import Iterator, Dict, List

import torch
from ortools.linear_solver import pywraplp
from torch.utils.data import IterableDataset

from data.datasets_base import MIPDataset
from data.mip_instance import MIPInstance
from metrics.general_metrics import Metrics
from metrics.mip_metrics import MIPMetrics, MIPMetrics_train
from utils.data_utils import MIPBatchHolder


class LPKnapsackDataset(MIPDataset, IterableDataset):
    """ Relaxed version of Knapsack
    """

    def __init__(self, min_variables, max_variables, max_copies=1, max_weight=10, max_values=10) -> None:
        super(LPKnapsackDataset, self).__init__()
        self._max_variables = max_variables
        self._min_variables = min_variables
        self._max_copies = max_copies
        self._max_weight = max_weight
        self._max_values = max_values

    def __iter__(self) -> Iterator[Dict]:
        def generator():
            while True:
                var_count = random.randint(self._min_variables, self._max_variables)
                var_indices = [i for i in range(var_count)]
                weights = [random.randint(1, self._max_weight) for _ in var_indices]

                values = [0] * len(var_indices)
                while sum(values) == 0:
                    values = [random.randint(0, self._max_values) for _ in var_indices]

                copies = [random.randint(1, self._max_copies) for _ in var_indices]

                max_weight = sum([w * c for w, c in zip(weights, copies)])
                min_weight = min(weights)
                capacity = random.randint(min_weight, max_weight)

                lp_solution, relaxed_int, relaxed_solution = self.relaxed_solutions(var_indices, weights, values, capacity)

                ip = self.convert_to_mip(var_indices, weights, values, copies, capacity)
                ip.optimal_solution_vars(var_indices, relaxed_solution)

                yield {"mip": ip,
                       "optimal_solution": torch.as_tensor([lp_solution], dtype=torch.float32)}

        return generator()

    def relaxed_solutions(self, var_indices, weights, values, capacity):
        solver = pywraplp.Solver.CreateSolver('GLOP')
        variables = [solver.NumVar(0, self._max_copies, str(i)) for i in var_indices]

        left_side = sum([w * v for w, v in zip(weights, variables)])
        solver.Add(left_side <= capacity)
        solver.Maximize(sum([v * var for v, var in zip(values, variables)]))

        solver.Solve()
        solution_vars = [v.solution_value() for v in variables]
        rounded_solutions = [round(s) for s in solution_vars]

        lp_solution = solver.Objective().Value()
        return -lp_solution, -sum([s * v for s, v in zip(rounded_solutions, values)]), [float(s) for s in solution_vars]

    def convert_to_mip(self, var_indices, weights, values, copies, capacity) -> MIPInstance:
        ip = MIPInstance(len(var_indices))

        # Weight less or equal with the knapsack capacity
        ip = ip.less_or_equal(var_indices, weights, capacity)
        ip = ip.maximize_objective(var_indices, values)

        return ip

    @property
    def required_output_bits(self):
        return self._max_copies.bit_length()

    def decode_model_outputs(self, model_output, batch_holder: MIPBatchHolder):
        return torch.squeeze(model_output)

    @property
    def test_metrics(self) -> List[Metrics]:
        return [MIPMetrics()]

    @property
    def train_metrics(self) -> List[Metrics]:
        return [MIPMetrics_train()]
