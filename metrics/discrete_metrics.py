import torch

from metrics.general_metrics import AverageMetrics, StackableMetrics


class DiscretizationMetrics(StackableMetrics):
    def __init__(self) -> None:
        super().__init__()
        self._avg = AverageMetrics()

    def update(self, prediction: torch.Tensor, **kwargs) -> None:
        self._avg.update(
            discrete_vs_continuous=self._discrete_vs_continuous(prediction),
            discrete_variables=self._count_discrete_variables(prediction),
            max_diff_to_discrete=self._max_diff_to_discrete(prediction),
        )

    @property
    def result(self):
        return self._avg.result

    @property
    def numpy_result(self):
        return self._avg.numpy_result

    def _count_discrete_variables(self, prediction: torch.Tensor):
        masked_vars = self._mask_discrete_variables(prediction)
        return torch.sum(masked_vars)

    def _discrete_vs_continuous(self, prediction: torch.Tensor):
        masked_vars = self._mask_discrete_variables(prediction)
        return torch.mean(masked_vars)

    @staticmethod
    def _mask_discrete_variables(prediction: torch.Tensor):
        values = torch.round(prediction) - prediction
        values = torch.abs(values)
        return torch.isclose(values, torch.zeros_like(values)).float()

    @staticmethod
    def _max_diff_to_discrete(prediction: torch.Tensor):
        values = torch.round(prediction) - prediction
        values = torch.abs(values)
        return torch.max(values)
