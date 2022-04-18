import torch

from metrics.general_metrics import AverageMetrics, StackableMetrics
from utils.data_utils import MIPBatchHolder


class SudokuMetrics(StackableMetrics):

    def __init__(self) -> None:
        super().__init__()
        self._avg = AverageMetrics()

    def update(self, prediction: torch.Tensor, batch_holder: MIPBatchHolder, **kwargs):
        givens, solution = batch_holder.get_data("givens", "labels")

        self._avg.update(
            rows_acc=self._rows_accuracy(prediction),
            col_acc=self._columns_accuracy(prediction),
            givens_acc=self._givens_accuracy(prediction, givens),
            range_acc=self._range_accuracy(prediction),
            square_acc=self._subsquare_accuracy(prediction),
            full_acc=self._full_accuracy(prediction, solution)
        )

    @property
    def result(self):
        return self._avg.result

    @property
    def numpy_result(self):
        return self._avg.numpy_result

    @staticmethod
    def _rows_accuracy(inputs: torch.Tensor):
        """
         Expect 3D tensor with dimensions [batch_size, 9, 9] with integer values
        """

        batch, r, c = inputs.size()
        result = torch.ones([batch, r], device=inputs.device)
        for i in range(1, 10, 1):
            value = torch.sum(torch.eq(inputs, i).int(), dim=-1)
            value = torch.clamp(value, 0, 1)
            result = result * value

        result = torch.mean(result.float(), dim=-1)
        return torch.mean(result)

    @staticmethod
    def _columns_accuracy(inputs):
        """
        Expect 3D tensor with dimensions [batch_size, 9, 9] with integer values
        """
        batch, r, c = inputs.size()
        result = torch.ones([batch, r], device=inputs.device)
        for i in range(1, 10, 1):
            value = torch.sum(torch.eq(inputs, i).int(), dim=-2)
            value = torch.clamp(value, 0, 1)
            result = result * value

        result = torch.mean(result.float(), dim=-1)
        return torch.mean(result)

    @staticmethod
    def _subsquare_accuracy(inputs):

        batch, r, c = inputs.size()
        squares = []
        for p in range(0, 3):
            for q in range(0, 3):
                subsquare = inputs[:, p * 3:(1 + p) * 3, q * 3:(1 + q) * 3]
                squares.append(torch.reshape(subsquare, [batch, r]))

        squares = torch.stack(squares, dim=-2)
        result = torch.ones([batch, r], device=inputs.device)

        for i in range(1, 10, 1):
            value = torch.sum(torch.eq(squares, i).int(), dim=-1)
            value = torch.clamp(value, 0, 1)
            result = result * value

        result = torch.mean(result.float(), dim=-1)
        return torch.mean(result)

    @staticmethod
    def _givens_accuracy(inputs, givens):
        mask = torch.clamp(givens, 0, 1)
        el_equal = torch.eq(mask * inputs, givens) * mask
        per_batch = torch.sum(el_equal.float(), dim=[-2, -1]) / torch.sum(mask, dim=[-2, -1])
        return torch.mean(per_batch)

    @staticmethod
    def _range_accuracy(inputs):
        geq = torch.greater_equal(inputs, 1)
        leq = torch.less_equal(inputs, 9)
        result = torch.logical_and(geq, leq)
        return torch.mean(torch.mean(result.float(), dim=[-2, -1]))

    @staticmethod
    def _full_accuracy(inputs, label):
        equal = torch.eq(inputs, label)
        equal = torch.all(equal, dim=-1)
        equal = torch.all(equal, dim=-1)
        return torch.mean(equal.float())
