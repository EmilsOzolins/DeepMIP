from abc import abstractmethod, ABC
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

import hyperparams as params
from data.datasets_base import MIPDataset
from data.ip_instance import IPInstance
from metrics.general_metrics import Metrics
from metrics.sudoku_metrics import SudokuMetrics


class IPSudokuDataset(MIPDataset, Dataset, ABC):

    def __init__(self, csv_file) -> None:
        csv = pd.read_csv(csv_file, header=0)
        self.features = csv["quizzes"]
        self.labels = csv["solutions"]
        self._sudoku_metrics = SudokuMetrics()

    def __getitem__(self, index) -> Dict:
        x = self.features[index]
        givens = self._prepare_sudoku(x)
        label = self.labels[index]
        label = self._prepare_sudoku(label)

        py_givens = givens
        givens = torch.as_tensor(givens)
        givens = torch.reshape(givens, [9, 9])

        label = torch.as_tensor(label)
        label = torch.reshape(label, [9, 9])

        return {"mip": self.prepare_mip(py_givens),
                "givens": givens,
                "labels": label}

    @staticmethod
    def _prepare_sudoku(data):
        return [int(c) for c in data]

    def __len__(self):
        return len(self.features)

    @abstractmethod
    def prepare_mip(self, data):
        pass

    @abstractmethod
    def decode_model_outputs(self, binary_assignment, decimal_assignment):
        pass

    @property
    def test_metrics(self) -> List[Metrics]:
        return [SudokuMetrics()]

    @property
    def train_metrics(self) -> List[Metrics]:
        return []


class BinarySudokuDataset(IPSudokuDataset):

    def prepare_mip(self, data):
        ip_inst = IPInstance(variable_count=9 ** 3)

        # Only one variable should be set to 1 along the k-dimension
        for i in range(9):
            for j in range(9):
                value = data[i + 9 * j]
                if value != 0:  # Set this element to given value, rest should be 0 in this field
                    ip_inst = ip_inst.equal([self._calc_index(i, j, value - 1)], [1], 1)
                    variables = [self._calc_index(i, j, k) for k in range(9) if k != value - 1]
                    multipliers = [1] * len(variables)
                    ip_inst = ip_inst.equal(variables, multipliers, 0)
                else:  # Only one element should be set to 1
                    variables = [self._calc_index(i, j, k) for k in range(9)]
                    multipliers = [1] * len(variables)
                    ip_inst = ip_inst.equal(variables, multipliers, 1)

        # All elements in single column should be different
        for j in range(9):
            for k in range(9):
                variables = [self._calc_index(i, j, k) for i in range(9)]
                multipliers = [1] * len(variables)
                ip_inst = ip_inst.equal(variables, multipliers, 1)

        # All elements in single row should be different
        for i in range(9):
            for k in range(9):
                variables = [self._calc_index(i, j, k) for j in range(9)]
                multipliers = [1] * len(variables)
                ip_inst = ip_inst.equal(variables, multipliers, 1)

        # All elements in each 3x3 sub-square should be different
        for p in range(3):
            for q in range(3):
                for k in range(9):
                    variables = [self._calc_index(i, j, k)
                                 for j in range(3 * p, 3 * (p + 1))
                                 for i in range(3 * q, 3 * (q + 1))
                                 ]
                    multipliers = [1] * len(variables)
                    ip_inst = ip_inst.equal(variables, multipliers, 1)

        return ip_inst

    @staticmethod
    def _calc_index(x, y, z) -> int:
        """ Indexes 3D tensor as 1D array, indexing should matches PyTorch reshape operation.
        """
        return 9 * x + 81 * y + z

    @property
    def required_output_bits(self):
        return 1

    def decode_model_outputs(self, binary_assignment, decimal_assignment):
        assignment = binary_assignment
        assignment = torch.round(assignment)  # TODO: I should round binary representation
        assignment = torch.reshape(assignment, [params.batch_size, 9, 9, 9])
        return torch.argmax(assignment, dim=-1) + 1


class IntegerSudokuDataset(IPSudokuDataset):

    def prepare_mip(self, data):
        ip_inst = IPInstance(variable_count=9 ** 2)

        # Elements should be in the range 1-9 and given values should be set
        for x in range(9):
            for y in range(9):
                index = self._calc_index(x, y)
                if data[index] == 0:
                    ip_inst.less_or_equal([index], [1], 9)
                    ip_inst.greater_or_equal([index], [1], 1)
                else:
                    ip_inst.equal([index], [1], data[index])

        big_m = 10
        next_free_var = 9 ** 2

        # Big-M reformulation of OR constraint
        # Original: x <= y-1 OR y <= x-1
        # Reformulated: x <= y-1 + M*(1-s1);
        #               y <= x-1 + M*(1-s2);
        #               s1+s2 = 1, where s1, s2 in {0,1}

        # All elements in single row should be different
        for x in range(9):
            for i in range(9):
                for j in range(i + 1, 9):
                    index_1 = self._calc_index(x, i)
                    index_2 = self._calc_index(x, j)

                    s1 = next_free_var
                    next_free_var += 1
                    s2 = next_free_var
                    next_free_var += 1

                    ip_inst.less_or_equal([index_1, index_2, s1], [1, -1, big_m], -1 + big_m)
                    ip_inst.less_or_equal([index_2, index_1, s2], [1, -1, big_m], -1 + big_m)
                    ip_inst.equal([s1, s2], [1, 1], 1)

        # All elements in single column should be different
        for y in range(9):
            for i in range(9):
                for j in range(i + 1, 9):
                    index_1 = self._calc_index(i, y)
                    index_2 = self._calc_index(j, y)

                    s1 = next_free_var
                    next_free_var += 1
                    s2 = next_free_var
                    next_free_var += 1

                    ip_inst.less_or_equal([index_1, index_2, s1], [1, -1, big_m], -1 + big_m)
                    ip_inst.less_or_equal([index_2, index_1, s2], [1, -1, big_m], -1 + big_m)
                    ip_inst.equal([s1, s2], [1, 1], 1)

            # All elements in each 3x3 sub-square should be different
            for p in range(3):
                for q in range(3):
                    variables = [self._calc_index(i, j)
                                 for j in range(3 * p, 3 * (p + 1))
                                 for i in range(3 * q, 3 * (q + 1))
                                 ]
                    for i, index_1 in enumerate(variables):
                        for index_2 in variables[i + 1:]:
                            s1 = next_free_var
                            next_free_var += 1
                            s2 = next_free_var
                            next_free_var += 1

                            ip_inst.less_or_equal([index_1, index_2, s1], [1, -1, big_m], -1 + big_m)
                            ip_inst.less_or_equal([index_2, index_1, s2], [1, -1, big_m], -1 + big_m)
                            ip_inst.equal([s1, s2], [1, 1], 1)

        return ip_inst

    @property
    def required_output_bits(self):
        return 4

    def decode_model_outputs(self, binary_assignment, decimal_assignment):
        powers = torch.tensor([2 ** k for k in range(self.required_output_bits)], dtype=torch.float32,
                              device=binary_assignment.device)
        assignment = torch.reshape(binary_assignment, [params.batch_size, 9, 9, self.required_output_bits])
        assignment = torch.round(assignment)
        return torch.sum(assignment * powers, dim=-1)

    @staticmethod
    def _calc_index(x, y) -> int:
        """ Indexes 3D tensor as 1D array, indexing should matches PyTorch reshape operation.
        """
        return 9 * x + y
