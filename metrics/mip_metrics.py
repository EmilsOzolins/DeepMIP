import torch

from metrics.general_metrics import AverageMetrics, StackableMetrics
from utils.data_utils import MIPBatchHolder, sparse_func


class MIPMetrics(StackableMetrics):

    def __init__(self) -> None:
        super().__init__()
        self._avg = AverageMetrics()

    def update(self, prediction: torch.Tensor, batch_holder: MIPBatchHolder, **kwargs):
        vars_const_graph = batch_holder.vars_const_graph
        const_values = batch_holder.const_values
        const_inst_graph = batch_holder.const_inst_graph
        vars_eq_const_graph = batch_holder.vars_eq_const_graph
        eq_const_values = batch_holder.eq_const_values
        eq_const_inst_graph = batch_holder.eq_const_inst_graph

        sat_const = self._satisfied_constraints(prediction, vars_const_graph, const_values)
        sat_eq_const = self._satisfied_eq_constraints(prediction, vars_eq_const_graph, eq_const_values)

        max_violation = self._max_constraints(prediction, vars_const_graph, const_values)
        mean_violation = self._max_constraints(prediction, vars_const_graph, const_values, aggregation_func=torch.mean)

        fully_sat_mips = self._mask_satisfied_instances(const_inst_graph, const_values, prediction, vars_const_graph)
        fully_sat_eq_mips = self._mask_satisfied_eq_instances(eq_const_inst_graph, eq_const_values, prediction, vars_eq_const_graph)
        fully_sat_mips_inst = torch.logical_and(fully_sat_mips, fully_sat_eq_mips)
        fully_sat_mips = torch.mean(fully_sat_mips_inst.float())

        vars_obj_graph = batch_holder.vars_obj_graph
        opt_value = batch_holder.optimal_solution

        mean_optimality_gap = self._mean_optimality_gap(vars_obj_graph, opt_value, prediction)
        max_optimality_gap = self._max_optimality_gap(vars_obj_graph, opt_value, prediction)

        found_optimum = self._count_optimal_values(opt_value, prediction, vars_obj_graph)
        fully_solved = torch.logical_and(fully_sat_mips_inst, found_optimum)
        fully_solved = torch.mean(fully_solved.float())

        self._avg.update(
            satisfied_constraints=sat_const,
            satisfied_eq_constraints=sat_eq_const,
            fully_satisfied_instances=fully_sat_mips,
            mean_optimality_gap=mean_optimality_gap,
            max_optimality_gap=max_optimality_gap,
            optimum_found=torch.mean(found_optimum.float()),
            fully_solved=fully_solved,
            max_violation=max_violation,
            mean_violation=mean_violation
        )

    @property
    def result(self):
        return self._avg.result

    @property
    def numpy_result(self):
        return self._avg.numpy_result

    @staticmethod
    def _count_optimal_values(opt_values, prediction, vars_obj_graph):
        predicted_val = torch.sparse.mm(vars_obj_graph.t(), torch.unsqueeze(prediction, dim=-1))
        found_optimum = torch.isclose(predicted_val, opt_values)
        return torch.squeeze(found_optimum)

    @staticmethod
    def _mean_optimality_gap(vars_obj_graph, opt_values, prediction):
        predicted_val = torch.sparse.mm(vars_obj_graph.t(), torch.unsqueeze(prediction, dim=-1))
        optimality_gap = predicted_val - opt_values
        return torch.mean(optimality_gap)

    @staticmethod
    def _median_optimality_gap(vars_obj_graph, opt_values, prediction):
        predicted_val = torch.sparse.mm(vars_obj_graph.t(), torch.unsqueeze(prediction, dim=-1))
        optimality_gap = predicted_val - opt_values
        return torch.median(optimality_gap)

    @staticmethod
    def _quantile_optimality_gap(q, vars_obj_graph, opt_values, prediction):
        predicted_val = torch.sparse.mm(vars_obj_graph.t(), torch.unsqueeze(prediction, dim=-1))
        optimality_gap = predicted_val - opt_values
        return torch.quantile(optimality_gap, q)

    @staticmethod
    def _max_optimality_gap(vars_obj_graph, opt_values, prediction):
        predicted_val = torch.sparse.mm(vars_obj_graph.t(), torch.unsqueeze(prediction, dim=-1))
        optimality_gap = predicted_val - opt_values
        return torch.max(optimality_gap)

    def _satisfied_constraints(self, prediction, vars_const_graph, const_values):
        sat_const = self._mask_sat_constraints(const_values, prediction, vars_const_graph)
        return torch.mean(sat_const)

    def _satisfied_eq_constraints(self, prediction, vars_eq_const_graph, eq_const_values):
        sat_const = self._mask_eq_sat_constraints(eq_const_values, prediction, vars_eq_const_graph)
        return torch.mean(sat_const)

    def _max_constraints(self, prediction, vars_const_graph, const_values, aggregation_func=torch.max):
        sat_const = self._max_constraint_violation(const_values, prediction, vars_const_graph,
                                                   aggregation_func=aggregation_func)
        return sat_const

    @staticmethod
    def _mask_sat_constraints(const_values, prediction, vars_const_graph, eps=1e-4):
        const_left_val = torch.sparse.mm(vars_const_graph.t(), torch.unsqueeze(prediction, dim=-1))
        abs_graph = sparse_func(vars_const_graph, torch.abs)
        scalers = torch.sparse.mm(abs_graph.t(), torch.unsqueeze(prediction,
                                                                 dim=-1))  # eps is proportional to the sum of all variable mulipliers in the constraint
        sat_const = torch.less_equal(const_left_val, torch.unsqueeze(const_values, dim=-1) + eps * scalers)
        sat_const = sat_const.float()
        return sat_const

    @staticmethod
    def _mask_eq_sat_constraints(const_values, prediction, vars_const_graph, eps=1e-4):
        const_left_val = torch.sparse.mm(vars_const_graph.t(), torch.unsqueeze(prediction, dim=-1))
        sat_const = torch.isclose(const_left_val, torch.unsqueeze(const_values, dim=-1), atol=eps)
        return sat_const.float()

    @staticmethod
    def _max_constraint_violation(const_values, prediction, vars_const_graph, aggregation_func=torch.max):
        const_left_val = torch.sparse.mm(vars_const_graph.t(), torch.unsqueeze(prediction, dim=-1))
        sat_const = aggregation_func(torch.relu(const_left_val - torch.unsqueeze(const_values, dim=-1)))
        sat_const = sat_const.float()
        return sat_const

    def _mask_satisfied_eq_instances(self, eq_const_inst_graph, eq_const_values, prediction, vars_eq_const_graph):
        sat_const = self._mask_eq_sat_constraints(eq_const_values, prediction, vars_eq_const_graph)

        sat_in_inst = torch.sparse.mm(eq_const_inst_graph.t(), sat_const).int()
        constraint_count = torch.sparse.sum(eq_const_inst_graph, dim=0).to_dense().int()
        sat_instances = torch.eq(torch.squeeze(sat_in_inst), constraint_count)

        return sat_instances

    def _mask_satisfied_instances(self, const_inst_graph, const_values, prediction, vars_const_graph):
        sat_const = self._mask_sat_constraints(const_values, prediction, vars_const_graph)

        sat_in_inst = torch.sparse.mm(const_inst_graph.t(), sat_const).int()
        constraint_count = torch.sparse.sum(const_inst_graph, dim=0).to_dense().int()
        sat_instances = torch.eq(torch.squeeze(sat_in_inst), constraint_count)

        return sat_instances


class MIPMetrics_train(MIPMetrics):
    def __init__(self) -> None:
        super().__init__()
        self._avg = AverageMetrics()

    def update(self, prediction: torch.Tensor, batch_holder: MIPBatchHolder, logits, **kwargs):
        logits = logits[:, 0]  # not rounded values

        const_inst_graph = batch_holder.const_inst_graph
        vars_const_graph = batch_holder.vars_const_graph
        const_values = batch_holder.const_values

        eq_const_inst_graph = batch_holder.eq_const_inst_graph
        eq_const_values = batch_holder.eq_const_values
        vars_eq_const_graph = batch_holder.vars_eq_const_graph

        max_violation = self._max_constraints(logits, vars_const_graph, const_values)
        mean_violation = self._max_constraints(logits, vars_const_graph, const_values, aggregation_func=torch.mean)
        sat_const = self._satisfied_constraints(logits, vars_const_graph, const_values)
        sat_eq_const = self._satisfied_eq_constraints(prediction, vars_eq_const_graph, eq_const_values)

        fully_sat_mips = self._mask_satisfied_instances(const_inst_graph, const_values, logits, vars_const_graph)
        fully_sat_eq_mips = self._mask_satisfied_eq_instances(eq_const_inst_graph, eq_const_values, prediction, vars_eq_const_graph)
        fully_sat_mips = torch.logical_and(fully_sat_mips, fully_sat_eq_mips).float()
        fully_sat_mips = torch.mean(fully_sat_mips)

        self._avg.update(
            satisfied_constraints=sat_const,
            sat_eq_const=sat_eq_const,
            fully_satisfied_instances=fully_sat_mips,
            max_violation=max_violation,
            mean_violation=mean_violation
        )
