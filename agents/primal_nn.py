from functools import cached_property, lru_cache
from typing import Any, List

import mip
import numpy as np
import pyscipopt
import torch
from mip import Model, xsum

import config as config
import hyperparams as params
from data.item_placement import ItemPlacementDataset
from data.load_balancing import LoadBalancingDataset
from data.mip_instance import MIPInstance
from model.mip_network import MIPNetwork
from utils.data_utils import InputDataHolder

def extract_current_ip_instance(model) -> MIPInstance:
    m = model.as_pyscipopt()  # type: pyscipopt.scip.Model

    variables = m.getVars(transformed=True)  # type: List[pyscipopt.scip.Variable]
    var2id = {var.name: idx for idx, var in enumerate(variables)}

    constraints = m.getLPRowsData()  # type: List[pyscipopt.scip.Row]
    constraints = [c for c in constraints if not c.isRemovable()]
    # TODO: Filter ones where all variables are set by bounds

    ip = MIPInstance()

    for const in constraints:
        c_mul = const.getVals()
        columns = const.getCols()  # type: List[pyscipopt.scip.Column]

        all_vars_set = all([c.getLb() == c.getUb() for c in columns])

        if all_vars_set:
            continue

        c_vars = [c.getVar() for c in columns]
        c_var_ids = [var2id[v.name] for v in c_vars]

        lhs = const.getLhs()
        rhs = const.getRhs()

        lhs_infinity = m.isInfinity(abs(lhs))
        rhs_infinity = m.isInfinity(abs(rhs))

        if lhs == rhs:
            ip.equal(c_var_ids, c_mul, lhs)
        elif not lhs_infinity:
            ip.greater_or_equal(c_var_ids, c_mul, lhs)
        elif not rhs_infinity:
            ip.less_or_equal(c_var_ids, c_mul, rhs)
        else:
            raise RuntimeError("Something is wrong with left or right side values!")

    sense = m.getObjectiveSense()
    objective = m.getObjective()  # type: pyscipopt.scip.Expr
    obj_exp = [(m.getTransformedVar(term.vartuple[0]), mul) for term, mul in objective.terms.items()]
    obj_exp = [(var2id[var.name], mul) for var, mul in obj_exp]

    objective_indices = [x for x, _ in obj_exp]
    objective_mul = [float(m) for _, m in obj_exp]

    if sense == "minimize":
        ip.minimize_objective(objective_indices, objective_mul)
    elif sense == "maximize":
        ip.maximize_objective(objective_indices, objective_mul)
    else:
        raise RuntimeError(f"Unknown sense direction! Expected 'maximize/minimize' but found {sense}")

    int_vars = [var2id[var.name] for var in variables if var.vtype() in {"BINARY", "INTEGER"}]
    ip.integer_constraint(int_vars)

    for var in variables:
        ip.variable_lower_bound(var2id[var.name], var.getLbLocal())
        ip.variable_lower_bound(var2id[var.name], var.getLbLocal())

    return ip


class Instance2Holder(InputDataHolder):

    def __init__(self, ip: MIPInstance, device: torch.device, **other_data) -> None:
        super().__init__()
        self._ip = ip
        self._device = device
        self._other_data = other_data

    @cached_property
    def vars_const_graph(self) -> torch.sparse.Tensor:
        indices = self._ip.variables_constraints_graph
        values = self._ip.variables_constraints_values
        size = self._ip.variables_constraints_graph_size

        torch_graph = torch.sparse_coo_tensor(indices, values,
                                              size, dtype=torch.float32,
                                              device=self._device)
        return torch_graph.coalesce()

    @cached_property
    def binary_vars_const_graph(self) -> torch.sparse.Tensor:
        indices = self._ip.variables_constraints_graph
        values = self._ip.variables_constraints_values
        size = self._ip.variables_constraints_graph_size

        torch_graph = torch.sparse_coo_tensor(indices,
                                              torch.ones_like(values), size,
                                              dtype=torch.float32, device=self._device)
        return torch_graph.coalesce()

    @cached_property
    def const_values(self) -> torch.Tensor:
        rhs = self._ip.right_values_of_constraints
        return torch.as_tensor(rhs, device=self._device, dtype=torch.float32)

    @cached_property
    def vars_obj_graph(self) -> torch.sparse.Tensor:
        indices = self._ip.variables_objective_graph
        values = self._ip.variables_objective_graph_values
        var_count, _ = self._ip.variables_constraints_graph_size

        return torch.sparse_coo_tensor(indices, values, size=[var_count, 1], device=self._device).coalesce()

    @cached_property
    def const_inst_graph(self) -> torch.sparse.Tensor:
        indices = self._ip.constraints_instance_graph
        values = self._ip.constraints_instance_graph_values
        _, const_count = self._ip.variables_constraints_graph_size

        return torch.sparse_coo_tensor(indices, values, size=[const_count, 1], device=self._device).coalesce()

    @cached_property
    def vars_inst_graph(self) -> torch.sparse.Tensor:
        indices = self._ip.variables_instance_graph
        values = self._ip.variables_instance_graph_values
        var_count, _ = self._ip.variables_constraints_graph_size

        return torch.sparse_coo_tensor(indices, values, size=[var_count, 1], device=self._device).coalesce()

    @cached_property
    def optimal_solution(self) -> torch.Tensor:
        return torch.tensor([float("nan")], device=self._device, dtype=torch.float32)

    @cached_property
    def integer_mask(self) -> torch.Tensor:
        var_count, const_count = self._ip.variables_constraints_graph_size
        mask = torch.zeros([var_count], device=self._device)
        indices = self._ip.integer_variables.to(device=self._device)
        return torch.scatter(mask, dim=0, index=indices, value=1.0)

    @cached_property
    def objective_multipliers(self) -> torch.Tensor:
        var_count = self.vars_obj_graph.size(0)

        if self.vars_obj_graph._nnz() == 0:
            return torch.zeros([var_count], device=self._device)

        return torch.sparse.sum(self.vars_obj_graph, dim=-1).to_dense()

    @lru_cache(maxsize=None)
    def get_data(self, *keys: str) -> Any:
        data = [self._other_data[k] for k in keys]
        data = [x.to(device=self._device) if isinstance(x, torch.Tensor) else x for x in data]

        return data if len(data) > 1 else data[0]


class NetworkPolicy():
    def __init__(self, problem):
        # called once for each problem benchmark
        self.rng = np.random.RandomState()
        self.problem = problem  # to devise problem-specific policies

        self.device = torch.device(config.device)
        self.network = MIPNetwork(
            output_bits=params.output_bits,
            feature_maps=params.feature_maps,
            pass_steps=params.recurrent_steps,
            summary=None
        )
        run_name = "20210927-155836"
        checkpoint = torch.load(f"/host-dir/mip_models/{run_name}/model.pth")

        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.network.eval()
        self.network.to(self.device)

        self.dataset = ItemPlacementDataset(
            "/host-dir/") if self.problem == "item_placement" else LoadBalancingDataset("/host-dir/")

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        # self.rng = np.random.RandomState(seed) # TODO: Enable random
        return 0

    def __call__(self, action_set, ip: MIPInstance):
        obs_holder = Instance2Holder(ip, self.device)

        with torch.no_grad():
            outputs, logits = self.network.forward(obs_holder, self.device)
        output = self.dataset.decode_model_outputs(outputs[-1], obs_holder)

        return action_set, output.cpu().numpy()[np.int64(action_set)]
