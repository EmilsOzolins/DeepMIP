import pandas as pd
import torch
from torch.utils.data.dataset import Dataset, T_co

from data.integer_programming_instance import IPInstance


class BinarySudokuDataset(Dataset):

    def __init__(self, csv_file) -> None:
        csv = pd.read_csv(csv_file, header=0)
        self.features = csv["quizzes"]
        self.labels = csv["solutions"]

    def __getitem__(self, index) -> T_co:
        x = self.features[index]
        givens = self._prepare_sudoku(x)
        label = self.labels[index]
        label = self._prepare_sudoku(label)

        return self._prepare_mip(givens), torch.as_tensor(givens), torch.as_tensor(label)

    @staticmethod
    def _prepare_sudoku(data):
        return [int(c) for c in data]

    def _prepare_mip(self, data):
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

    def __len__(self):
        return len(self.features)
