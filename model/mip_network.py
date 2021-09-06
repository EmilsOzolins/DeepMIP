import torch
import torch.nn as nn

from model.normalization import PairNorm
from utils.data import MIPBatchHolder


class MIPNetwork(torch.nn.Module):

    def __init__(self, output_bits, feature_maps=64, pass_steps=3):
        super().__init__()

        self.feature_maps = feature_maps
        self.pass_steps = pass_steps

        self.constraint_update = nn.Sequential(
            nn.Linear(self.feature_maps * 2, self.feature_maps),
            nn.ReLU(),
            nn.Linear(self.feature_maps, self.feature_maps),
            PairNorm()
        )

        self.make_query_constraints = nn.Sequential(
            nn.Linear(self.feature_maps, self.feature_maps),
            nn.ReLU(),
            nn.Linear(self.feature_maps, self.feature_maps),
        )

        self.make_query_objective = nn.Sequential(
            nn.Linear(self.feature_maps, self.feature_maps),
            nn.ReLU(),
            nn.Linear(self.feature_maps, self.feature_maps),
        )

        self.variable_update = nn.Sequential(
            nn.Linear(self.feature_maps * 3, self.feature_maps),
            nn.ReLU(),
            nn.Linear(self.feature_maps, self.feature_maps),
            PairNorm()
        )

        self.output = nn.Sequential(
            nn.Linear(self.feature_maps, self.feature_maps),
            nn.ReLU(),
            nn.Linear(self.feature_maps, output_bits)
        )

        self.noise = torch.distributions.Normal(0, 1)

        self.step = 0
        self.powers_of_two = torch.as_tensor([2 ** k for k in range(0, output_bits)], dtype=torch.float32,
                                             device=torch.device('cuda:0'))

    def forward(self, batch_holder: MIPBatchHolder, device):
        var_count, const_count = batch_holder.vars_const_graph.size()
        _, objective_count = batch_holder.vars_inst_graph.size()

        variables = torch.ones([var_count, self.feature_maps], device=device)
        constraints = torch.ones([const_count, self.feature_maps], device=device)

        binary_outputs = []
        decimal_outputs = []

        obj_multipliers = torch.sparse.sum(batch_holder.vars_obj_graph, dim=-1).to_dense()
        obj_multipliers = torch.unsqueeze(obj_multipliers, dim=-1)

        const_values = torch.unsqueeze(batch_holder.const_values, dim=-1)

        for i in range(self.pass_steps):
            const_query = self.make_query_constraints(variables)  # TODO: Experiment with noise
            var2const_msg = torch.sparse.mm(batch_holder.vars_const_graph.t(), const_query)
            const_loss = torch.relu(var2const_msg - const_values)

            const_msg = torch.cat([constraints, const_loss], dim=-1)
            constraints = self.constraint_update(const_msg)

            obj_query = self.make_query_objective(variables)
            obj_query = obj_query * obj_multipliers

            const2var_msg = torch.sparse.mm(batch_holder.vars_const_graph, constraints)
            var_msg = torch.cat([variables, const2var_msg, obj_query], dim=-1)
            variables = self.variable_update(var_msg)

            out_vars = self.output(variables)
            out = torch.sigmoid(out_vars + self.noise.sample(out_vars.size()).cuda())

            binary_outputs.append(out)

            decimal_pred = torch.sum(self.powers_of_two * out, dim=-1, keepdim=True)  # TODO: Get rid of this
            decimal_outputs.append(decimal_pred)

            constraints = constraints.detach() * 0.2 + constraints * 0.8
            variables = variables.detach() * 0.2 + variables * 0.8

        return binary_outputs, decimal_outputs
