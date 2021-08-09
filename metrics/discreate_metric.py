import torch

from metrics.average_metrics import AverageMetric


class DiscretizationMetric(AverageMetric):

    def __init__(self) -> None:
        super().__init__()

    def update(self, binary_prediction: torch.Tensor) -> None:
        super(DiscretizationMetric, self).update({
            "discrete_vs_continuous": self._discrete_vs_continuous(binary_prediction),
            "discrete_variables": self._count_discrete_variables(binary_prediction),
            "max_diff_to_discrete": self._max_diff_to_discrete(binary_prediction),
            "median_to_discrete": self._calculate_median(binary_prediction)
        })

    @staticmethod
    def _calculate_median(binary_prediction: torch.Tensor):
        return torch.mean(torch.median(binary_prediction, dim=-1))

    def _count_discrete_variables(self, binary_prediction: torch.Tensor):
        discrete_vars = self._count_discrete_per_instance(binary_prediction)
        return torch.mean(discrete_vars)

    def _discrete_vs_continuous(self, binary_prediction: torch.Tensor):
        _, total_variables = binary_prediction.size()
        discrete_vars = self._count_discrete_per_instance(binary_prediction)
        return torch.mean(discrete_vars / total_variables)

    @staticmethod
    def _count_discrete_per_instance(binary_prediction: torch.Tensor):
        values = torch.round(binary_prediction) - binary_prediction
        values = torch.abs(values)
        values = torch.eq(values, torch.zeros(values)).float()
        discrete_vars = torch.sum(values, dim=-1)
        return discrete_vars

    def _max_diff_to_discrete(self, binary_prediction: torch.Tensor):
        values = torch.round(binary_prediction) - binary_prediction
        values = torch.abs(values)
        values = torch.max(values, dim=-1)
        return torch.mean(values)
