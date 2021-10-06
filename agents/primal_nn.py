from functools import cached_property, lru_cache
from typing import Any, List

import numpy as np
import pyscipopt
import torch

import config as config
import hyperparams as params
from data.item_placement import ItemPlacementDataset
from data.load_balancing import LoadBalancingDataset
from data.mip_instance import MIPInstance
from main import sum_loss
from model.mip_network import MIPNetwork
from utils.data_utils import InputDataHolder


def extract_current_ip_instance(model: pyscipopt.scip.Model) -> MIPInstance:
    variables = model.getVars(transformed=True)  # type: List[pyscipopt.scip.Variable]
    var2id = {var.name: idx for idx, var in enumerate(variables)}

    constraints = model.getLPRowsData()  # type: List[pyscipopt.scip.Row]
    constraints = [c for c in constraints if not c.isRemovable()]

    ip = MIPInstance()

    for const in constraints:
        c_mul = const.getVals()
        columns = const.getCols()  # type: List[pyscipopt.scip.Column]
        c_vars = [model.getTransformedVar(c.getVar()) for c in columns]

        if all([v.getLbLocal() == v.getUbLocal() for v in c_vars]):  # Remove constraints that are already satisfied
            continue

        c_var_ids = [var2id[v.name] for v in c_vars]

        lhs = const.getLhs()
        rhs = const.getRhs()

        lhs_infinity = model.isInfinity(abs(lhs))
        rhs_infinity = model.isInfinity(abs(rhs))

        if lhs == rhs:
            ip.equal(c_var_ids, c_mul, lhs)
        elif not lhs_infinity:
            ip.greater_or_equal(c_var_ids, c_mul, lhs)
        elif not rhs_infinity:
            ip.less_or_equal(c_var_ids, c_mul, rhs)
        else:
            raise RuntimeError("Something is wrong with left or right side values!")

    sense = model.getObjectiveSense()
    objective = model.getObjective()  # type: pyscipopt.scip.Expr
    obj_exp = [(model.getTransformedVar(term.vartuple[0]), mul) for term, mul in objective.terms.items()]
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
        ip.variable_upper_bound(var2id[var.name], var.getUbLocal())

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
        run_name = "20211006-093510"
        checkpoint = torch.load(f"/host-dir/mip_models/{run_name}/model.pth")

        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.network.eval()
        self.network.to(self.device)

        self.dataset = ItemPlacementDataset(
            "/host-dir/") if self.problem == "item_placement" else LoadBalancingDataset("/host-dir/")

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        self.rng = np.random.RandomState(seed)

    def __call__(self, action_set, ip: MIPInstance):
        obs_holder = Instance2Holder(ip, self.device)

        with torch.no_grad():
            outputs, logits = self.network.forward(obs_holder, self.device)

        l, loss_c, loss_o, best_logit_map = sum_loss(outputs[-1:], obs_holder)
        output = self.dataset.decode_model_outputs(outputs[-1][:, best_logit_map:best_logit_map + 1], obs_holder)

        # TODO: Understand what to return
        left_const = torch.sparse.mm(obs_holder.vars_const_graph.t(), torch.unsqueeze(output, dim=-1))
        sat_const = torch.relu(torch.squeeze(left_const) - obs_holder.const_values)

        non_zero = sat_const != 0
        is_zero = sat_const == 0

        sat_const[non_zero] = 1
        sat_const[is_zero] = 0

        vars_in_sat_const = torch.sparse.mm(obs_holder.binary_vars_const_graph, torch.unsqueeze(sat_const, dim=-1))
        vars_in_sat_const = torch.squeeze(vars_in_sat_const).cpu().numpy()
        vars_in_sat_const = {i for i, x in enumerate(vars_in_sat_const) if x == 0}

        action_set = [x for x in action_set if x in vars_in_sat_const]

        return action_set, output.cpu().numpy()[np.int64(action_set)]
