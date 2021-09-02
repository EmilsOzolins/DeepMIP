import torch

from metrics.general_metrics import AverageMetrics, Metrics


class MIPMetrics(Metrics):

    def __init__(self) -> None:
        super().__init__()
        self._avg = AverageMetrics()

    def update(self, predictions, vars_const_graph, b_values, const_inst_graph, **kwargs):
        sat_const = self._satisfied_constraints(predictions, vars_const_graph, b_values, const_inst_graph)
        fully_sat_mips = self._fully_satisfied(predictions, vars_const_graph, b_values, const_inst_graph)

        self._avg.update(satisfied_constraints=sat_const, fully_satisfied_constraints=fully_sat_mips)

    @property
    def result(self):
        return self._avg.result

    @property
    def numpy_result(self):
        return self._avg.numpy_result

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
