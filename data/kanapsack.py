import random
from abc import abstractmethod
from typing import Iterator

from torch.utils.data import IterableDataset
from torch.utils.data._typing import T_co

from data.ip_instance import IPInstance


class BoundedKnapsackDataset(IterableDataset):
    """ Base for random Knapsack problem.
        For more information see: https://en.wikipedia.org/wiki/Knapsack_problem
    """

    def __init__(self, min_variables, max_variables, max_copies, max_weight=10, max_values=10) -> None:
        self._max_variables = max_variables
        self._min_variables = min_variables
        self._max_copies = max_copies
        self._max_weight = max_weight
        self._max_values = max_values

    def __iter__(self) -> Iterator[T_co]:
        def generator():
            var_count = random.randint(self._min_variables, self._max_variables)
            weights = [random.randint(0, self._max_weight) for _ in range(var_count)]
            values = [random.randint(0, self._max_values) for _ in range(var_count)]
            copies = [random.randint(1, self._max_copies) for _ in range(var_count)]
            var_indices = [i for i in range(var_count)]

            max_weight = sum([w * c for w, c in zip(weights, copies)])
            min_weight = sum(weights)
            capacity = random.randint(min_weight, max_weight)

            yield self.convert_to_mip(var_indices, weights, values, copies, capacity)

        return generator()

    @staticmethod
    def convert_to_mip(var_indices, weights, values, copies, capacity):
        ip = IPInstance(len(var_indices))

        for ind, c in zip(var_indices, copies):
            ip.less_or_equal([ind], [1], c)

        ip.less_or_equal(var_indices, weights, capacity)
        ip.objective_function(var_indices, values)

        return ip

    @property
    def required_output_bits(self):
        return self._max_copies.bit_length()

    @abstractmethod
    def decode_model_outputs(self, binary_assignment, decimal_assignment):
        pass


class BinaryKnapsackDataset(BoundedKnapsackDataset):

    def __init__(self, min_variables, max_variables, max_weight=10, max_values=10) -> None:
        super().__init__(min_variables, max_variables, 1, max_weight, max_values)