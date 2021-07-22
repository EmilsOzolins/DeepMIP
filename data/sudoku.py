import pandas as pd
import torch
from torch.utils.data.dataset import Dataset, T_co


class SudokuDataset(Dataset):

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

    @staticmethod
    def _prepare_mip(data, dtype=torch.float32):
        indices = []
        a_values = []

        high_value = 9
        low_value = 1

        const_index = 0
        b_values = []

        # Should be in range 1-9 and set given numbers
        for i, v in enumerate(data):
            indices.append((i, const_index))
            a_values.append(1)
            b_values.append(high_value if v == 0 else v)
            const_index += 1

        for i, v in enumerate(data):
            indices.append((i, const_index))
            a_values.append(-1)
            b_values.append(-low_value if v == 0 else -v)
            const_index += 1

        new_variable = len(data)

        # This should encode that first rows elements are not equal
        for i in range(9):
            for j in range(i + 1, 9, 1):
                indices.append((j, const_index))
                a_values.append(1)
                indices.append((i, const_index))
                a_values.append(-1)
                indices.append((new_variable, const_index))
                a_values.append(9)
                b_values.append(9 - 1)

                const_index += 1
                new_variable += 1

                indices.append((i, const_index))
                a_values.append(1)
                indices.append((j, const_index))
                a_values.append(-1)
                indices.append((new_variable, const_index))
                a_values.append(-9)
                b_values.append(-1)

                const_index += 1

                indices.append((new_variable - 1, const_index))
                a_values.append(1)
                indices.append((new_variable, const_index))
                a_values.append(1)
                b_values.append(1)

                const_index += 1

                indices.append((new_variable - 1, const_index))
                a_values.append(-1)
                indices.append((new_variable, const_index))
                a_values.append(-1)
                b_values.append(-1)

                new_variable += 1
                const_index += 1

        i = [x for x, _ in indices]
        j = [x for _, x in indices]

        adj_matrix = torch.sparse_coo_tensor(torch.tensor([i, j]), torch.tensor(a_values), dtype=dtype,
                                             device=torch.device('cuda:0'))
        return adj_matrix, torch.tensor(b_values, dtype=dtype, device=torch.device('cuda:0'))

    def __len__(self):
        return len(self.features)
