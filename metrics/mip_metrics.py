import torch

from metrics.average_metrics import AverageMetric


class MIPMetrics(AverageMetric):

    def __init__(self) -> None:
        super().__init__()

    def update(self, predictions, vars_const_graph, b_values, const_inst_graph):
        super(MIPMetrics, self).update(
            {"satisfied_constraints": self._satisfied_constraints(predictions, vars_const_graph, b_values,
                                                                  const_inst_graph),
             "fully_satisfied_mips": self._fully_satisfied(predictions, vars_const_graph, b_values, const_inst_graph)

             }
        )

    def _satisfied_constraints(self, predictions, constraints_adj_matrix, b_values, const_inst_graph):
        const_count, sat_per_inst = self._sat_and_total_constraints(b_values, const_inst_graph,
                                                                    constraints_adj_matrix, predictions)

        return torch.mean(sat_per_inst / const_count)

    def _fully_satisfied(self, predictions, constraints_adj_matrix, b_values, const_inst_graph):
        const_count, sat_per_inst = self._sat_and_total_constraints(b_values, const_inst_graph,
                                                                    constraints_adj_matrix, predictions)

        sat_instances = torch.eq(sat_per_inst, const_count).float()
        return torch.mean(sat_instances)

    @staticmethod
    def _sat_and_total_constraints(b_values, const_inst_graph, constraints_adj_matrix, predictions):
        results = torch.sparse.mm(constraints_adj_matrix.t(), torch.unsqueeze(predictions, dim=-1))
        satisfied = torch.less_equal(results, torch.unsqueeze(b_values, dim=-1)).float()
        sat_per_inst = torch.squeeze(torch.sparse.mm(const_inst_graph.t(), satisfied))
        const_count = torch.sparse.sum(const_inst_graph, dim=0).to_dense()
        return const_count, sat_per_inst
