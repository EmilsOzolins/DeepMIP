import random
from abc import abstractmethod
from typing import Iterator, Dict, List

import torch
from ortools.algorithms import pywrapknapsack_solver
from torch.utils.data import IterableDataset

from data.datasets_base import MIPDataset
from data.mip_instance import MIPInstance
from metrics.general_metrics import Metrics
from metrics.mip_metrics import MIPMetrics


class BoundedKnapsackDataset(MIPDataset, IterableDataset):
    """ Base for random Knapsack problem.
        For more information see: https://en.wikipedia.org/wiki/Knapsack_problem
    """

    def __init__(self, min_variables, max_variables, max_copies, max_weight=10, max_values=10) -> None:
        super(BoundedKnapsackDataset, self).__init__()
        self._max_variables = max_variables
        self._min_variables = min_variables
        self._max_copies = max_copies
        self._max_weight = max_weight
        self._max_values = max_values
        self._metrics_knapsack = None
        self._average_metrics = None

    def __iter__(self) -> Iterator[Dict]:
        def generator():
            while True:
                var_count = random.randint(self._min_variables, self._max_variables)
                var_indices = [i for i in range(var_count)]
                weights = [random.randint(1, self._max_weight) for _ in var_indices]
                values = [random.randint(0, self._max_values) for _ in var_indices]
                copies = [random.randint(1, self._max_copies) for _ in var_indices]

                max_weight = sum([w * c for w, c in zip(weights, copies)])
                min_weight = min(weights)
                capacity = random.randint(min_weight, max_weight)

                solution = self.get_optimal_value(weights, values, [capacity])
                yield {"mip": self.convert_to_mip(var_indices, weights, values, copies, capacity),
                       "optimal_solution": torch.as_tensor([solution], dtype=torch.float32)}

        return generator()

    @abstractmethod
    def get_optimal_value(self, weights, values, capacities):
        pass

    def convert_to_mip(self, var_indices, weights, values, copies, capacity):
        ip = MIPInstance(len(var_indices))

        # Each element is available c times
        for ind, c in zip(var_indices, copies):
            ip = ip.less_or_equal([ind], [1], c)

        # Weight less or equal with the knapsack capacity
        ip = ip.less_or_equal(var_indices, weights, capacity)
        ip = ip.integer_constraint(var_indices)
        ip = ip.maximize_objective(var_indices, values)

        return ip

    @property
    def required_output_bits(self):
        return self._max_copies.bit_length()

    @abstractmethod
    def decode_model_outputs(self, model_output):
        pass

    @property
    def test_metrics(self) -> List[Metrics]:
        return [MIPMetrics()]

    @property
    def train_metrics(self) -> List[Metrics]:
        return []


class BinaryKnapsackDataset(BoundedKnapsackDataset):

    def __init__(self, min_variables, max_variables, max_weight=20, max_values=20) -> None:
        super().__init__(min_variables, max_variables, max_copies=1, max_weight=max_weight, max_values=max_values)

    def decode_model_outputs(self, model_output):
        assignments = torch.round(model_output)
        return torch.squeeze(assignments)

    def get_optimal_value(self, weights, values, capacities):
        solver = pywrapknapsack_solver.KnapsackSolver(
            pywrapknapsack_solver.KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER, 'KnapsackExample')

        solver.Init(values, [weights], capacities)
        return solver.Solve()

    def convert_to_mip(self, var_indices, weights, values, copies, capacity):
        ip = MIPInstance(len(var_indices))

        ip = ip.less_or_equal(var_indices, weights, capacity)
        ip = ip.maximize_objective(var_indices, values)
        ip = ip.integer_constraint(var_indices)

        return ip

# TODO: Unbounded Knapsack dataset

# TODO: Add hard instances from http://hjemmesider.diku.dk/~pisinger/codes.html
