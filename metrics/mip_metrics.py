import torch

from metrics.general_metrics import AverageMetrics, Metrics


class MIPMetrics(Metrics):

    # TODO: Deal with duplications
    def __init__(self) -> None:
        super().__init__()
        self._avg = AverageMetrics()

    def update(self, predictions, vars_const_graph, const_values, const_inst_graph, vars_obj_graph, opt_value, **kwargs):
        sat_const = self._satisfied_constraints(predictions, vars_const_graph, const_values, const_inst_graph)
        fully_sat_mips = self._fully_satisfied(predictions, vars_const_graph, const_values, const_inst_graph)

        mean_optimality_gap = self._mean_optimality_gap(vars_obj_graph, opt_value, predictions)
        max_optimality_gap = self._max_optimality_gap(vars_obj_graph, opt_value, predictions)

        found_optimum = self._found_optimum(vars_obj_graph, opt_value, predictions)
        fully_solved = self._totally_solved(vars_const_graph, const_inst_graph, const_values, vars_obj_graph, opt_value, predictions)

        self._avg.update(
            satisfied_constraints=sat_const,
            fully_satisfied_instances=fully_sat_mips,
            mean_optimality_gap=mean_optimality_gap,
            max_optimality_gap=max_optimality_gap,
            optimum_found=found_optimum,
            fully_solved=fully_solved
        )

    @property
    def result(self):
        return self._avg.result

    @property
    def numpy_result(self):
        return self._avg.numpy_result

    @staticmethod
    def _found_optimum(vars_obj_graph, opt_values, prediction):
        predicted_val = torch.sparse.mm(vars_obj_graph.t(), torch.unsqueeze(prediction, dim=-1))
        predicted_val = torch.abs(predicted_val)

        found_optimum = torch.isclose(predicted_val, opt_values)
        found_optimum = torch.squeeze(found_optimum)
        found_optimum = found_optimum.float()

        return torch.mean(found_optimum)

    def _totally_solved(self, vars_const_graph, const_inst_graph, const_values, vars_obj_graph, opt_values, prediction):
        results = torch.sparse.mm(vars_const_graph.t(), torch.unsqueeze(prediction, dim=-1))
        satisfied = torch.less_equal(results, torch.unsqueeze(const_values, dim=-1)).float()
        sat_per_inst = torch.squeeze(torch.sparse.mm(const_inst_graph.t(), satisfied))
        const_count = torch.sparse.sum(const_inst_graph, dim=0).to_dense()
        sat_instances = torch.eq(sat_per_inst, const_count)

        predicted_val = torch.sparse.mm(vars_obj_graph.t(), torch.unsqueeze(prediction, dim=-1))
        predicted_val = torch.abs(predicted_val)

        found_optimum = torch.isclose(predicted_val, opt_values)
        found_optimum = torch.squeeze(found_optimum)
        found_optimum = found_optimum.float()

        fully_solved = torch.logical_and(sat_instances, found_optimum).float()
        return torch.mean(fully_solved)

    @staticmethod
    def _mean_optimality_gap(vars_obj_graph, opt_values, prediction):
        predicted_val = torch.sparse.mm(vars_obj_graph.t(), torch.unsqueeze(prediction, dim=-1))
        predicted_val = torch.abs(predicted_val)
        optimality_gap = torch.abs(opt_values - predicted_val)

        return torch.mean(optimality_gap)

    @staticmethod
    def _max_optimality_gap(vars_obj_graph, opt_values, prediction):
        predicted_val = torch.sparse.mm(vars_obj_graph.t(), torch.unsqueeze(prediction, dim=-1))
        predicted_val = torch.abs(predicted_val)
        optimality_gap = torch.abs(opt_values - predicted_val)

        return torch.max(optimality_gap)

    def _satisfied_constraints(self, predictions, constraints_adj_matrix, b_values, const_inst_graph):
        const_count, sat_per_inst = self._sat_and_total_constraints(b_values, const_inst_graph, constraints_adj_matrix, predictions)
        return torch.mean(sat_per_inst / const_count) # TODO: They calculate the same value, fix this

    def _fully_satisfied(self, predictions, constraints_adj_matrix, b_values, const_inst_graph):
        const_count, sat_per_inst = self._sat_and_total_constraints(b_values, const_inst_graph, constraints_adj_matrix, predictions)
        sat_instances = torch.eq(sat_per_inst, const_count).float()
        return torch.mean(sat_instances)

    @staticmethod
    def _sat_and_total_constraints(const_values, const_inst_graph, vars_const_graph, predictions):
        const_left_val = torch.sparse.mm(vars_const_graph.t(), torch.unsqueeze(predictions, dim=-1))
        satisfied = torch.less_equal(const_left_val, torch.unsqueeze(const_values, dim=-1)).float()

        sat_per_inst = torch.squeeze(torch.sparse.mm(const_inst_graph.t(), satisfied))
        const_count = torch.sparse.sum(const_inst_graph, dim=0).to_dense()

        return const_count, sat_per_inst
