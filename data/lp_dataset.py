import glob
import multiprocessing as mp
import random
import warnings
from typing import List, Dict

import mip
import torch
from mip import Model, OptimizationStatus
from torch.utils.data import Dataset

import config
from data.datasets_base import MIPDataset
from data.mip_instance import MIPInstance
from metrics.general_metrics import Metrics
from metrics.mip_metrics import MIPMetrics, MIPMetrics_train
from utils.data_utils import InputDataHolder


class LPDataset(MIPDataset, Dataset):
    """
    WARNING: Files are cached on disk! If any changes are made cache should be deleted manually.
    """

    def __init__(self, data_folder, find_solutions=False,
                 augment: bool = False, augment_with_objective: bool = False) -> None:
        self._instances = glob.glob(data_folder + "/*.lp")
        self._should_augment = augment
        self._find_solutions = find_solutions
        self._augment_objective = augment_with_objective

    def prefill_cache(self):
        # Prefills cache in a sequential manner
        inst_count = len(self._instances)
        step = inst_count // mp.cpu_count()

        arguments = [(start, start + step, step) for start in range(0, inst_count, step)]
        pool = mp.Pool(mp.cpu_count())
        pool.starmap_async(self._prefill_cache_subprocess, arguments)
        pool.close()
        pool.join()

    def _prefill_cache_subprocess(self, start, end, step_size):
        processed_instances = 0
        for idx in range(start, end, 1):
            self.__getitem__(idx)  # Just to populate cache

            if processed_instances % 1000 == 0:
                print(f"Processed {processed_instances} instances from {step_size} total")
            processed_instances += 1

    @property
    def required_output_bits(self):
        return 1

    @property
    def test_metrics(self) -> List[Metrics]:
        return [MIPMetrics()]

    @property
    def train_metrics(self) -> List[Metrics]:
        return [MIPMetrics_train()]

    def decode_model_outputs(self, model_output, batch_holder: InputDataHolder):
        output = torch.squeeze(model_output)
        mask = batch_holder.integer_mask
        inv_mask = 1 - mask
        return torch.round(output * mask) + output * inv_mask

    def __getitem__(self, index: int) -> Dict:
        file_name = self._instances[index]
        item = get_item(file_name, self._should_augment, self._find_solutions)

        if self._augment_objective:
            # TODO: Handle optimization sense
            item["mip"] = item["mip"].less_or_equal(item["mip"]._objective_indices,
                                                    item["mip"]._objective_multipliers,
                                                    random.random() * 800)

        return item

    def __len__(self) -> int:
        return len(self._instances)

    def __add__(self, other: 'LPDataset') -> 'LPDataset':
        self._instances += other._instances
        return self


@config.cache.memoize()
def get_item(file_name, should_augment, find_solutions):
    ip = get_mip_instance(file_name, find_solutions=find_solutions)
    return {"mip": ip.augment() if should_augment else ip,
            "optimal_solution": ip.objective_value}


def get_mip_instance(file_name: str, find_solutions=False):
    try:
        model = Model()
        model.read(file_name)

        # vars_in_prob = set()
        # vars_in_prob.update(model.objective.expr.keys())
        # for const in model.constrs:
        #     const_exp = const.expr  # type: mip.LinExpr
        #     vars_in_prob.update(const_exp.expr.keys())
        #
        # variables = model.vars
        # variables_not_in_prob = set(variables).difference(vars_in_prob)
        # model.remove(list(variables_not_in_prob))  # Make instances smaller by removing redundant variables

        variables = model.vars

        variable_count = len(variables)
        variable_map = {var: idx for idx, var in enumerate(variables)}

        ip = MIPInstance(variable_count)

        for const in model.constrs:
            const_exp = const.expr  # type: mip.LinExpr

            var_indices = [variable_map[var] for var in const_exp.expr.keys()]
            coefficients = [float(c) for c in const_exp.expr.values()]
            rhs = const.rhs

            if const_exp.sense == "=":
                ip.equal(var_indices, coefficients, rhs)
            elif const_exp.sense == "<":
                ip.less_or_equal(var_indices, coefficients, rhs)
            elif const_exp.sense == ">":
                ip.greater_or_equal(var_indices, coefficients, rhs)
            else:
                raise RuntimeError("Constraint sense not found! Please check your MIP file.")

        objective = model.objective

        var_indices = [variable_map[var] for var in objective.expr.keys()]
        coefficients = [float(c) for c in objective.expr.values()]

        if model.sense == 'MIN':
            ip = ip.minimize_objective(var_indices, coefficients)
            optimization_sign = 1
        elif model.sense == 'MAX':
            ip = ip.maximize_objective(var_indices, coefficients)
            optimization_sign = -1  # reverse the optimum value for maximization tasks
        else:
            raise RuntimeError("Model sense not found! Please check your MIP file.")

        int_vars = [var for var in variables if var.var_type in {'B', 'I'}]
        integer_vars = [variable_map[var] for var in int_vars]
        ip = ip.integer_constraint(integer_vars)

        int_vars = set(int_vars)
        for var in variables:
            if var in int_vars and var.lb == 0 and var.ub == 1:
                continue  # Don't include integer variables
            ip = ip.variable_lower_bound(variable_map[var], var.lb)
            ip = ip.variable_upper_bound(variable_map[var], var.ub)

        if find_solutions:
            model.preprocess = 0
            model.verbose = 0
            model.emphasis = 1  # prioritize feasible solutions
            status = model.optimize(max_seconds=2)

            if status in {OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE}:
                obj_value = model.objective_value * optimization_sign
            else:
                warnings.warn(f"Solution not found in the time limit,"
                              f" will use 0 as objective. Returned status was {status}")
                obj_value = 0
            ip = ip.presolved_objective_value(float(obj_value))
        else:
            ip = ip.presolved_objective_value(0)
    except Exception as ex:
        raise Exception(f"Please delete {file_name}") from ex

    model.preprocess = 0
    model.verbose = 0
    model.emphasis = 1
    status = model.optimize(max_seconds=2, relax=True)
    if status in {OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE}:
        for var in variables:
            var = var  # type:mip.Var
            ip.variable_relaxed_solution(variable_map[var], var.x)
    else:
        warnings.warn(f"Relaxed solution not found in the time limit. Returned status was {status}")

    return ip
