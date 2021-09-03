import torch

from metrics.general_metrics import AverageMetrics, Metrics


class MIPMetrics(Metrics):

    def __init__(self) -> None:
        super().__init__()
        self._avg = AverageMetrics()

    def update(self, predictions, vars_const_graph, const_values, const_inst_graph, vars_obj_graph, opt_value, **kwargs):
        sat_const = self._satisfied_constraints(predictions, vars_const_graph, const_values)
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

    def _found_optimum(self, vars_obj_graph, opt_values, prediction):
        found_optimum = self._count_optimal_values(opt_values, prediction, vars_obj_graph)
        return torch.mean(found_optimum.float())

    def _totally_solved(self, vars_const_graph, const_inst_graph, const_values, vars_obj_graph, opt_values, prediction):
        sat_instances = self._mask_satisfied_instances(const_inst_graph, const_values, prediction, vars_const_graph)
        found_optimum = self._count_optimal_values(opt_values, prediction, vars_obj_graph)

        fully_solved = torch.logical_and(sat_instances, found_optimum)

        return torch.mean(fully_solved.float())

    @staticmethod
    def _count_optimal_values(opt_values, prediction, vars_obj_graph):
        predicted_val = torch.sparse.mm(vars_obj_graph.t(), torch.unsqueeze(prediction, dim=-1))
        predicted_val = torch.abs(predicted_val)

        found_optimum = torch.isclose(predicted_val, opt_values)
        return torch.squeeze(found_optimum)

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

    def _satisfied_constraints(self, prediction, vars_const_graph, const_values):
        sat_const = self._mask_sat_constraints(const_values, prediction, vars_const_graph)
        return torch.mean(sat_const)

    @staticmethod
    def _mask_sat_constraints(const_values, prediction, vars_const_graph):
        const_left_val = torch.sparse.mm(vars_const_graph.t(), torch.unsqueeze(prediction, dim=-1))
        sat_const = torch.less_equal(const_left_val, torch.unsqueeze(const_values, dim=-1))
        sat_const = sat_const.float()
        return sat_const

    def _fully_satisfied(self, prediction, vars_const_graph, const_values, const_inst_graph):
        sat_instances = self._mask_satisfied_instances(const_inst_graph, const_values, prediction, vars_const_graph)
        return torch.mean(sat_instances.float())

    def _mask_satisfied_instances(self, const_inst_graph, const_values, prediction, vars_const_graph):
        sat_const = self._mask_sat_constraints(const_values, prediction, vars_const_graph)

        sat_in_inst = torch.sparse.mm(const_inst_graph.t(), sat_const).int()
        constraint_count = torch.sparse.sum(const_inst_graph, dim=0).to_dense().int()
        sat_instances = torch.eq(torch.squeeze(sat_in_inst), constraint_count)

        return sat_instances
