from abc import abstractmethod
from collections import defaultdict
from functools import cached_property, lru_cache
from typing import Tuple, List, Dict, Any

import torch
from data.mip_instance import MIPInstance
from torch import Tensor


def batch_data(batch: List[Dict]):
    """Data batching with support for batching sparse adjacency matrices"""
    batch_data = defaultdict(list)

    for data in batch:
        for key, item in data.items():
            batch_data[key].append(item)

    output_batch = dict()
    for key, data in batch_data.items():
        if isinstance(data[0], MIPInstance):
            output_batch[key] = batch_as_mip(data)
            continue

        if isinstance(data[0], Tensor):
            output_batch[key] = batch_as_tensor(data)
            continue

        if isinstance(data[0], int):
            output_batch[key] = torch.as_tensor(data)
            continue

        raise NotImplementedError(f"Batching for {type(data[0])} is not implemented!")

    return output_batch


def batch_as_mip(mip_instances: Tuple[MIPInstance]) -> Dict[str, Tuple]:
    """ Batches mip instances as one huge sparse graph.
    Graphs should be converted to sparse tensor after data is collected by workers.
    """
    var_const_edge_indices = []  # Main MIP graph as variables-constraints adjacency matrix
    var_const_edge_values = []
    constrain_right_values = []  # Values in the right side of inequalities
    var_offset = 0
    const_offset = 0
    size = [0, 0]

    objective_edge_indices = []  # Objective value as variables-instance adjacency matrix
    objective_edge_values = []

    const_edge_indices = []  # Constraints-instances adjacency matrix (servers as helper)
    const_edge_values = []

    variables_edge_indices = []  # Variables-instances adjacency matrix (servers as helper)
    variables_edge_values = []

    integer_variables = []

    for graph_id, mip in enumerate(mip_instances):
        ind = mip.variables_constraints_graph
        ind[0, :] += var_offset
        ind[1, :] += const_offset

        size[0] += mip.variables_constraints_graph_size[0]
        size[1] += mip.variables_constraints_graph_size[1]

        var_const_edge_indices.append(ind)
        var_const_edge_values.append(mip.variables_constraints_values)
        constrain_right_values.append(mip.right_values_of_constraints)

        obj_ind = mip.variables_objective_graph
        obj_ind[0, :] += var_offset
        obj_ind[1, :] += graph_id

        objective_edge_indices.append(obj_ind)
        objective_edge_values.append(mip.variables_objective_graph_values)

        const_ind = mip.constraints_instance_graph
        const_ind[0, :] += const_offset
        const_ind[1, :] += graph_id

        const_edge_indices.append(const_ind)
        const_edge_values.append(mip.constraints_instance_graph_values)

        var_ind = mip.variables_instance_graph
        var_ind[0, :] += var_offset
        var_ind[1, :] += graph_id

        variables_edge_indices.append(var_ind)
        variables_edge_values.append(mip.variables_instance_graph_values)

        int_vars = mip.integer_variables
        int_vars += var_offset
        integer_variables.append(int_vars)

        var_offset += mip.next_variable_index
        const_offset += mip.next_constraint_index

    var_const_edge_indices = torch.cat(var_const_edge_indices, dim=-1)
    var_const_edge_values = torch.cat(var_const_edge_values, dim=-1)
    constrain_right_values = torch.cat(constrain_right_values, dim=-1)
    constraints = var_const_edge_indices, var_const_edge_values, constrain_right_values, size

    batch_size = len(mip_instances)
    objective_edge_indices = torch.cat(objective_edge_indices, dim=-1)
    objective_edge_values = torch.cat(objective_edge_values, dim=-1)
    objective = objective_edge_indices, objective_edge_values, [size[0], batch_size]

    const_edge_indices = torch.cat(const_edge_indices, dim=-1)
    const_edge_values = torch.cat(const_edge_values, dim=-1)
    consts_per_graph = const_edge_indices, const_edge_values, [size[1], batch_size]

    variables_edge_indices = torch.cat(variables_edge_indices, dim=-1)
    variables_edge_values = torch.cat(variables_edge_values, dim=-1)
    vars_per_graph = variables_edge_indices, variables_edge_values, [size[0], batch_size]

    integer_variables = torch.cat(integer_variables, dim=-1)

    return {"constraints": constraints,
            "objective": objective,
            "consts_per_graph": consts_per_graph,
            "vars_per_graph": vars_per_graph,
            "integer_variables": integer_variables}


def batch_as_tensor(batch_data: Tuple[Tensor]):
    return torch.stack(batch_data, dim=0)


class InputDataHolder:

    @property
    @abstractmethod
    def vars_const_graph(self) -> torch.sparse.Tensor:
        pass

    @property
    @abstractmethod
    def binary_vars_const_graph(self) -> torch.sparse.Tensor:
        pass

    @property
    @abstractmethod
    def const_values(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def vars_obj_graph(self) -> torch.sparse.Tensor:
        pass

    @property
    @abstractmethod
    def const_inst_graph(self) -> torch.sparse.Tensor:
        pass

    @property
    @abstractmethod
    def vars_inst_graph(self) -> torch.sparse.Tensor:
        pass

    @property
    @abstractmethod
    def optimal_solution(self) -> torch.Tensor:
        """ Returns precomputed optimal value of objective function.
        """
        pass

    @property
    @abstractmethod
    def integer_mask(self) -> torch.Tensor:
        """ Returns mask over integer variables where 1 if variable should be integer and 0 for others. """
        pass

    @property
    @abstractmethod
    def objective_multipliers(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_data(self, *keys: str) -> Any:
        """ Here you can get various task specific data, that is not directly related to MIP.
        """
        pass


class MIPBatchHolder(InputDataHolder):

    def __init__(self, batched_data: dict, device) -> None:
        self._batched_data = batched_data
        self._device = device

    @cached_property
    def vars_const_graph(self):
        indices, values, _, size = self._batched_data["mip"]["constraints"]
        return torch.sparse_coo_tensor(indices, values, size=size, device=self._device).coalesce()

    @cached_property
    def binary_vars_const_graph(self):
        indices, values, _, size = self._batched_data["mip"]["constraints"]
        return torch.sparse_coo_tensor(indices, torch.ones_like(values), size=size, device=self._device)

    @cached_property
    def const_values(self):
        *_, constraint_values, _ = self._batched_data["mip"]["constraints"]
        return constraint_values.to(device=self._device)

    @cached_property
    def vars_obj_graph(self):
        indices, values, size = self._batched_data["mip"]["objective"]
        return torch.sparse_coo_tensor(indices, values, size=size, device=self._device).coalesce()

    @cached_property
    def const_inst_graph(self):
        indices, values, size = self._batched_data["mip"]["consts_per_graph"]
        return torch.sparse_coo_tensor(indices, values, size=size, device=self._device).coalesce()

    @cached_property
    def vars_inst_graph(self):
        indices, values, size = self._batched_data["mip"]["vars_per_graph"]
        return torch.sparse_coo_tensor(indices, values, size=size, device=self._device).coalesce()

    @cached_property
    def optimal_solution(self):
        return self._batched_data['optimal_solution'].to(device=self._device)

    @cached_property
    def integer_mask(self):
        *_, size = self._batched_data["mip"]["vars_per_graph"]
        mask = torch.zeros([size[0]], device=self._device)
        indices = self._batched_data["mip"]["integer_variables"].to(device=self._device)
        return torch.scatter(mask, dim=0, index=indices, value=1.0)

    @cached_property
    def objective_multipliers(self):
        var_count = self.vars_obj_graph.size(0)

        if self.vars_obj_graph._nnz() == 0:
            return torch.zeros([var_count], device=self._device)

        return torch.sparse.sum(self.vars_obj_graph, dim=-1).to_dense()

    @lru_cache(maxsize=None)
    def get_data(self, *keys: str):
        data = [self._batched_data[k] for k in keys]
        data = [x.to(device=self._device) if isinstance(x, torch.Tensor) else x for x in data]

        return data if len(data) > 1 else data[0]


# applies the given function to sparse tensor values
def sparse_func(vars_obj_graph, func):
    abs_graph = torch.sparse_coo_tensor(vars_obj_graph.indices(),
                                        func(vars_obj_graph.values()),
                                        size=vars_obj_graph.size(),
                                        device=vars_obj_graph.device)
    return abs_graph


def make_sparse_unit(vars_obj_graph):
    abs_graph = torch.sparse_coo_tensor(vars_obj_graph.indices(),
                                        torch.ones_like(vars_obj_graph.values()),
                                        size=vars_obj_graph.size(),
                                        device=vars_obj_graph.device)
    return abs_graph

def ssqrt(x, alpha=1):
    """Signed sqrt
    larger alpha makes the function more non-linear
    """
    return x * torch.rsqrt(1 + alpha*torch.abs(x))

def sparse_dense_mul(s, d):
  i = s._indices()
  v = s._values()
  dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
  return torch.sparse.FloatTensor(i, v * dv, s.size())

def sparse_dense_mul_dim0(s, d):
  i = s._indices()
  v = s._values()
  dv = d[i[0,:]]  # get values from relevant entries of dense matrix
  return torch.sparse.FloatTensor(i, v * dv, s.size())

def sparse_dense_mul_dim1(s, d):
  i = s._indices()
  v = s._values()
  dv = d[i[0,:]]  # get values from relevant entries of dense matrix
  return torch.sparse.FloatTensor(i, v * dv, s.size())