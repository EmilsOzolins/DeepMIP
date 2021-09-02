import torch

from metrics.general_metrics import AverageMetrics, Metrics


class DiscretizationMetrics(Metrics):
    def __init__(self) -> None:
        super().__init__()
        self._avg = AverageMetrics()

    def update(self, binary_prediction: torch.Tensor, **kwargs) -> None:
        self._avg.update(
            discrete_vs_continuous=self._discrete_vs_continuous(binary_prediction),
            discrete_variables=self._count_discrete_variables(binary_prediction),
            max_diff_to_discrete=self._max_diff_to_discrete(binary_prediction),
        )

    @property
    def result(self):
        return self._avg.result

    @property
    def numpy_result(self):
        return self._avg.numpy_result

    def _count_discrete_variables(self, binary_prediction: torch.Tensor):
        discrete_vars = self._count_discrete_per_instance(binary_prediction)
        return torch.mean(discrete_vars)

    def _discrete_vs_continuous(self, binary_prediction: torch.Tensor):
        total_variables = binary_prediction.size()[0]
        discrete_vars = self._count_discrete_per_instance(binary_prediction)
        return torch.mean(discrete_vars / total_variables)

    @staticmethod
    def _count_discrete_per_instance(binary_prediction: torch.Tensor):
        values = torch.round(binary_prediction) - binary_prediction
        values = torch.abs(values)
        values = torch.eq(values, torch.zeros_like(values)).float()
        discrete_vars = torch.sum(values, dim=-1)
        return discrete_vars

    @staticmethod
    def _max_diff_to_discrete(binary_prediction: torch.Tensor):
        values = torch.round(binary_prediction) - binary_prediction
        values = torch.abs(values)
        values = torch.max(values, dim=-1).values
        return torch.mean(values)
