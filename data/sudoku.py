import pandas as pd
import torch
from torch.utils.data.dataset import Dataset, T_co


class SudokuDataset(Dataset):

    def __init__(self, csv_file) -> None:
        csv = pd.read_csv(csv_file, header=0, verbose=True)
        self.features = csv["quizzes"]
        self.labels = csv["solutions"]

    def __getitem__(self, index) -> T_co:
        x = self.features[index]
        x = self._prepare_sudoku(x)
        y = self.labels[index]
        y = self._prepare_sudoku(y)

        return self._prepare_mip(x), y

    @staticmethod
    def _prepare_sudoku(data, dtype=torch.int32):
        data = [int(c) for c in data]
        return torch.tensor(data, dtype=dtype)

    @staticmethod
    def _prepare_mip(data, dtype=torch.float32):
        indices = []
        a_values = []

        high_value = 9
        low_value = 1

        const_index = 0
        b_values = []

        for i, _ in enumerate(data):
            indices.append((i, const_index))
            const_index += 1
            a_values.append(1)
            b_values.append(high_value)

        for i, _ in enumerate(data):
            indices.append((i, const_index))
            const_index += 1
            a_values.append(-1)
            b_values.append(-low_value)

        i = [x for x, _ in indices]
        j = [x for _, x in indices]

        adj_matrix = torch.sparse_coo_tensor(torch.tensor([i, j]), torch.tensor(a_values), dtype=dtype)
        return adj_matrix, torch.tensor(b_values, dtype=dtype)

    def __len__(self):
        return len(self.features)
