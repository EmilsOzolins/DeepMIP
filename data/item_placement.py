import ctypes
import glob
import gzip
import multiprocessing as mp
import pickle
import warnings
from pathlib import Path
from typing import List, Dict

import mip
import numpy as np
import torch
from mip import Model, OptimizationStatus
from torch.utils.data import Dataset

from data.datasets_base import MIPDataset
from data.mip_instance import MIPInstance
from metrics.general_metrics import Metrics
from metrics.mip_metrics import MIPMetrics, MIPMetrics_train
from utils.data import InputDataHolder


class ItemPlacementDataset(MIPDataset, Dataset):
    """
    WARNING: Files are cached on disk! If any changes are made cache should be deleted manually.
    """

    def __init__(self, data_folder, augment: bool = False, cache_dir="/tmp/cache/item_placement") -> None:
        self._instances = glob.glob(data_folder + "/*.lp")
        self._should_augment = augment
        self._cache_dir = Path(cache_dir)

        if not self._cache_dir.exists():
            self._cache_dir.mkdir(parents=True)

        # Shared array between workers
        shared_array_base = mp.Array(ctypes.c_bool, len(self._instances))
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        self._in_cache = shared_array.reshape(len(self._instances))

        for idx in range(len(self._in_cache)):
            self._in_cache[idx] = False

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

        if self._in_cache[index]:
            with gzip.open(cached_file) as file:
                ip = pickle.load(file)
        elif cached_file.exists():
            self._in_cache[index] = True
            with gzip.open(cached_file) as file:
                ip = pickle.load(file)
        else:
            ip = self.get_mip_instance(file_name)
            with gzip.open(cached_file, mode="wb") as file:
                pickle.dump(ip, file)

            self._in_cache[index] = True

        return {"mip": ip.augment() if self._should_augment else ip,
                "optimal_solution": ip.objective_value}

    @staticmethod
    def get_mip_instance(file_name: str):
        model = Model()
        model.read(file_name)

        vars_in_prob = set()
        vars_in_prob.update(model.objective.expr.keys())
        for const in model.constrs:
            const_exp = const.expr  # type: mip.LinExpr
            vars_in_prob.update(const_exp.expr.keys())

        variables = model.vars
        variables_not_in_prob = set(variables).difference(vars_in_prob)
        model.remove(list(variables_not_in_prob))
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
        elif model.sense == 'MAX':
            ip = ip.maximize_objective(var_indices, coefficients)
        else:
            raise RuntimeError("Model sense not found! Please check your MIP file.")

        int_vars = [var for var in variables if var.var_type in {'B', 'I'}]
        real_vars = [var for var in variables if var.var_type == 'C']

        integer_vars = [variable_map[var] for var in int_vars]
        ip = ip.integer_constraint(integer_vars)

        for var in real_vars:
            ip = ip.less_or_equal([variable_map[var]], [1], var.ub)
            ip = ip.greater_or_equal([variable_map[var]], [1], var.lb)

        model.preprocess = 0
        model.verbose = 0
        model.emphasis = 1  # prioritize feasible solutions
        status = model.optimize(max_seconds=2)
        if status not in {OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE}:
            warnings.warn("Solution not found in the time limit, will use nan as objective.")
            ip = ip.presolved_objective_value(float('nan'))
        else:
            ip = ip.presolved_objective_value(float(model.objective_value))

        return ip

    def __len__(self) -> int:
        return len(self._instances)

    def __add__(self, other: 'ItemPlacementDataset') -> 'ItemPlacementDataset':
        self._instances += other._instances
        return self
