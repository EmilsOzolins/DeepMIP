import torch

from metrics.general_metrics import AverageMetrics, StackableMetrics


class DiscretizationMetrics(StackableMetrics):
    def __init__(self) -> None:
        super().__init__()
        self._avg = AverageMetrics()

    def update(self, logits: torch.Tensor, **kwargs) -> None:
        self._avg.update(
            discrete_fraction=self._discrete_vs_continuous(logits),
            max_diff_to_discrete=self._max_diff_to_discrete(logits),
        )

    @property
    def result(self):
        return self._avg.result

    @property
    def numpy_result(self):
        return self._avg.numpy_result

    def _count_discrete_variables(self, logits: torch.Tensor):
        masked_vars = self._mask_discrete_variables(logits)
        return torch.sum(masked_vars)

    def _discrete_vs_continuous(self, logits: torch.Tensor):
        masked_vars = self._mask_discrete_variables(logits)
        return torch.mean(masked_vars)

    @staticmethod
    def _mask_discrete_variables(logits: torch.Tensor):
        values = torch.round(logits) - logits
        values = torch.abs(values)
        return torch.isclose(values, torch.zeros_like(values), atol=0.01).float()

    @staticmethod
    def _max_diff_to_discrete(logits: torch.Tensor):
        values = torch.round(logits) - logits
        values = torch.abs(values)
        return torch.max(values)
