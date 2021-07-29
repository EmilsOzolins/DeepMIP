import pandas as pd
import torch
from torch.utils.data.dataset import Dataset, T_co

# TODO: Rewrite this using PyTorch geometric - it already has batching and stuff for graphs
from data.integer_programming_instance import IPInstanceBuilder


class BinarySudokuDataset(Dataset):

    def __init__(self, csv_file) -> None:
        csv = pd.read_csv(csv_file, header=0)
        self.features = csv["quizzes"]
        self.labels = csv["solutions"]

    def __getitem__(self, index) -> T_co:
        x = self.features[index]
        x = self._prepare_sudoku(x)
        y = self.labels[index]
        y = self._prepare_sudoku(y)

        return self._prepare_mip(x), x

    @staticmethod
    def _prepare_sudoku(data, dtype=torch.int32):
        data = [int(c) for c in data]
        return torch.tensor(data, dtype=dtype)

    def _prepare_mip(self, data):
        ip_inst = IPInstanceBuilder()

        # At least one number should be selected in the field
        for i in range(9):
            for j in range(9):
                variables = [self._calc_index(i, j, k) for k in range(9)]
                multipliers = [1] * len(variables)
                ip_inst = ip_inst.equal(variables, multipliers, 1)

        # Given elements should be equal to given value
        for i in range(9):
            for j in range(9):
                value = data[9 * i + j]
                if value != 0:
                    ip_inst = ip_inst.equal([self._calc_index(i, j, value - 1)], [1], 1)

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

        return ip_inst.create()

    @staticmethod
    def _calc_index(x, y, z) -> int:
        """ Indexes 3D tensor as 1D array, indexing should matches PyTorch reshape operation.
        """
        return 9 * x + y + 81 * z

    def __len__(self):
        return len(self.features)
