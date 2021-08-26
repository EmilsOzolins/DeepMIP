import torch

from metrics.average_metrics import AverageMetric


class KnapsackMetrics(AverageMetric):

    def __init__(self) -> None:
        super().__init__()

    def update(self, predictions, constraints_adj_matrix, b_values):
        super(KnapsackMetrics, self).update(
            {"satisfied_constraints": self._satisfied_constraints(predictions, constraints_adj_matrix, b_values),

             }
        )

    @staticmethod
    def _satisfied_constraints(predictions, constraints_adj_matrix, b_values):
        # TODO: This is MIP generic metric, move to mip_metrics.py ...
        # TODO: Calculate values per instance not per batch
        results = torch.sparse.mm(constraints_adj_matrix.t(), torch.unsqueeze(predictions, dim=-1))
        results = torch.squeeze(results)
        satisfied = torch.less_equal(results, b_values).int()
        _, const_count = constraints_adj_matrix.size()
        return torch.sum(satisfied) / const_count

    # TODO: Metric for optimality gap
