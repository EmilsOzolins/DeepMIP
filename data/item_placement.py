import glob
import gzip
import pickle
import warnings
from pathlib import Path
from typing import List, Dict

import mip
import torch
from mip import Model, OptimizationStatus
from torch.utils.data import Dataset

from data.datasets_base import MIPDataset
from data.mip_instance import MIPInstance
from metrics.general_metrics import Metrics
from metrics.mip_metrics import MIPMetrics, MIPMetrics_train
from utils.data_utils import InputDataHolder


class ItemPlacementDataset(MIPDataset, Dataset):
    """
    WARNING: Files are cached on disk! If any changes are made cache should be deleted manually.
    """

    def __init__(self, data_folder, find_solutions=False,
                 augment: bool = False, cache_dir="/tmp/cache/item_placement") -> None:
        self._instances = glob.glob(data_folder + "/*.lp")
        self._should_augment = augment
        self._cache_dir = Path(cache_dir)
        self._find_solutions = find_solutions

        if not self._cache_dir.exists():
            self._cache_dir.mkdir(parents=True)

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
        file_path = Path(file_name)
        name = file_path.name + ".pickle.gz"

        cached_file = self._cache_dir / name

        if cached_file.exists():
            with gzip.open(cached_file) as file:
                ip = pickle.load(file)
        else:
            ip = self.get_mip_instance(file_name)
            with gzip.open(cached_file, mode="wb") as file:
                pickle.dump(ip, file)

        return {"mip": ip.augment() if self._should_augment else ip,
                "optimal_solution": ip.objective_value}

    def get_mip_instance(self, file_name: str):
        try:
            model = Model()
            model.read(file_name)

            vars_in_prob = set()
            vars_in_prob.update(model.objective.expr.keys())
            for const in model.constrs:
                const_exp = const.expr  # type: mip.LinExpr
                vars_in_prob.update(const_exp.expr.keys())

            variables = model.vars
            variables_not_in_prob = set(variables).difference(vars_in_prob)
            model.remove(list(variables_not_in_prob))  # Make instances smaller by removing redundant variables
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
                optimization_sign = -1 # reverse the optimum value for maximization tasks
            else:
                raise RuntimeError("Model sense not found! Please check your MIP file.")

            int_vars = [var for var in variables if var.var_type in {'B', 'I'}]
            integer_vars = [variable_map[var] for var in int_vars]
            ip = ip.integer_constraint(integer_vars)

            for var in variables:
                ip = ip.variable_lower_bound(variable_map[var], var.lb)
                ip = ip.variable_upper_bound(variable_map[var], var.ub)

            if self._find_solutions:
                model.preprocess = 0
                model.verbose = 0
                model.emphasis = 1  # prioritize feasible solutions
                status = model.optimize(max_seconds=2)

                if status in {OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE}:
                    obj_value = model.objective_value * optimization_sign
                else:
                    warnings.warn(f"Solution not found in the time limit,"
                                  f" will use nan as objective. Return status was {status}")
                    obj_value = 'nan'
                ip.presolved_objective_value(float(obj_value))
            else:
                ip = ip.presolved_objective_value(float('nan'))
        except Exception as ex:
            raise Exception(f"Please delete {file_name}") from ex

        return ip

    def __len__(self) -> int:
        return len(self._instances)

    def __add__(self, other: 'ItemPlacementDataset') -> 'ItemPlacementDataset':
        self._instances += other._instances
        return self
