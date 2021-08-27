from metrics.average_metrics import AverageMetric


class KnapsackMetrics(AverageMetric):

    def __init__(self) -> None:
        super().__init__()

    def update(self, predictions, constraints_adj_matrix, b_values):
        super(KnapsackMetrics, self).update(
            {

            }
        )

    # TODO: Metric for optimality gap
