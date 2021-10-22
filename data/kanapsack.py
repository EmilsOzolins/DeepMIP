import random
import warnings
from abc import abstractmethod
from typing import Iterator, Dict, List

import torch
from ortools.algorithms import pywrapknapsack_solver
from ortools.linear_solver import pywraplp
from torch.utils.data import IterableDataset

from data.datasets_base import MIPDataset
from data.mip_instance import MIPInstance
from metrics.general_metrics import Metrics
from metrics.mip_metrics import MIPMetrics, MIPMetrics_train
from utils.data_utils import MIPBatchHolder


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
        self._skipped_instances = 0

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

                relaxed_int, relaxed_solution = self.relaxed_solutions(var_indices, weights, values, capacity)
                solution, solution_vars = self.get_optimal_value(weights, values, [capacity])

                if relaxed_int == solution:
                    # Don't include solutions that can be obtained from LP by rounding variables
                    self._skipped_instances += 1
                    if self._skipped_instances % 100 == 0:
                        warnings.warn(
                            f"Instances seems to be too easy, {self._skipped_instances} consecutive examples skipped.")
                    continue

                self._skipped_instances = 0

                ip = self.convert_to_mip(var_indices, weights, values, copies, capacity)
                for v_id, relax_val in zip(var_indices, relaxed_solution):
                    ip.variable_relaxed_solution(v_id, relax_val)

                ip.optimal_solution_vars(var_indices, solution_vars)

                yield {"mip": ip,
                       "optimal_solution": torch.as_tensor([solution], dtype=torch.float32)}

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

        return -sum([s * v for s, v in zip(rounded_solutions, values)]), [float(s) for s in solution_vars]

    @abstractmethod
    def get_optimal_value(self, weights, values, capacities):
        pass

    def convert_to_mip(self, var_indices, weights, values, copies, capacity) -> MIPInstance:
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
    def decode_model_outputs(self, model_output, batch_holder: MIPBatchHolder):
        pass

    @property
    def test_metrics(self) -> List[Metrics]:
        return [MIPMetrics()]

    @property
    def train_metrics(self) -> List[Metrics]:
        return [MIPMetrics_train()]


class BinaryKnapsackDataset(BoundedKnapsackDataset):

    def __init__(self, min_variables, max_variables, augment=False, max_weight=20, max_values=20) -> None:
        super().__init__(min_variables, max_variables, max_copies=1, max_weight=max_weight, max_values=max_values)
        self._augment = augment

    def decode_model_outputs(self, model_output, batch_holder: MIPBatchHolder):
        assignments = torch.round(model_output)
        return torch.squeeze(assignments)

    def get_optimal_value(self, weights, values, capacities):
        solver = pywrapknapsack_solver.KnapsackSolver(
            pywrapknapsack_solver.KnapsackSolver.KNAPSACK_BRUTE_FORCE_SOLVER, 'KnapsackExample')

        solver.Init(values, [weights], capacities)
        solution = -solver.Solve()
        solution_vars = [1 if solver.BestSolutionContains(idx) else 0 for idx in range(len(weights))]

        return solution, solution_vars

    def convert_to_mip(self, var_indices, weights, values, copies, capacity):
        ip = MIPInstance(len(var_indices))

        ip = ip.less_or_equal(var_indices, weights, capacity)
        ip = ip.maximize_objective(var_indices, values)
        ip = ip.integer_constraint(var_indices)

        return ip.augment() if self._augment else ip


class ConstrainedBinaryKnapsackDataset(BinaryKnapsackDataset):

    def __init__(self, min_variables, max_variables, max_weight=20, max_values=20) -> None:
        self.suboptimality = 3.0
        super().__init__(min_variables, max_variables, max_weight=max_weight, max_values=max_values)

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

                relaxed_int = self.relaxed_solutions(var_indices, weights, values, capacity)
                solution = self.get_optimal_value(weights, values, [capacity])

                if relaxed_int == solution:
                    # Don't include solutions that can be obtained from LP by rounding variables
                    continue

                yield {"mip": self.convert_to_mip_thr(var_indices, weights, values, copies, capacity,
                                                      solution - self.suboptimality),
                       "optimal_solution": torch.as_tensor([0.], dtype=torch.float32)}

        return generator()

    def convert_to_mip_thr(self, var_indices, weights, values, copies, capacity, objective_threshold):
        ip = MIPInstance(len(var_indices))

        ip = ip.less_or_equal(var_indices, weights, capacity)
        ip = ip.greater_or_equal(var_indices, values, objective_threshold)
        ip = ip.integer_constraint(var_indices)

        return ip

# TODO: Unbounded Knapsack dataset

# TODO: Add hard instances from http://hjemmesider.diku.dk/~pisinger/codes.html
