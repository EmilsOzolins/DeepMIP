import torch
import torch.nn as nn

from model.normalization import NodeNorm
from utils.data import MIPBatchHolder


class MIPNetwork(torch.nn.Module):

    def __init__(self, output_bits, feature_maps=64, pass_steps=3):
        super().__init__()

        self.feature_maps = feature_maps
        self.pass_steps = pass_steps

        self.constraint_update = nn.Sequential(
            nn.Linear(self.feature_maps * 3, self.feature_maps),
            NodeNorm(),
            nn.ReLU(),
            nn.Linear(self.feature_maps, self.feature_maps * 2),
        )

        self.make_query = nn.Sequential(
            nn.Linear(self.feature_maps, self.feature_maps),
            nn.ReLU(),
            nn.Linear(self.feature_maps, self.feature_maps),
        )

        self.variable_update = nn.Sequential(
            nn.Linear(self.feature_maps * 4 + 1, self.feature_maps),
            NodeNorm(),
            nn.ReLU(),
            nn.Linear(self.feature_maps, self.feature_maps),
        )

        self.output = nn.Sequential(
            nn.Linear(self.feature_maps, self.feature_maps),
            NodeNorm(),
            nn.ReLU(),
            nn.Linear(self.feature_maps, output_bits)
        )

        self.noise = torch.distributions.Normal(0, 1)

        self.step = 0

    def forward(self, batch_holder: MIPBatchHolder, device):
        # TODO: Experiment with disentangled architecture
        var_count, const_count = batch_holder.vars_const_graph.size()
        _, objective_count = batch_holder.vars_inst_graph.size()

        variables = torch.ones([var_count, self.feature_maps], device=device)
        constraints = torch.ones([const_count, self.feature_maps], device=device)

        outputs = []

        const_values = torch.unsqueeze(batch_holder.const_values, dim=-1)
        obj_multipliers = torch.unsqueeze(batch_holder.objective_multipliers, dim=-1)

        # TODO: Experiment with mean
        const_scaler = torch.sparse.sum(batch_holder.vars_const_graph, dim=0).to_dense()
        const_scaler = torch.unsqueeze(const_scaler, dim=-1)

        # TODO: Experiment with mean
        vars_scaler = torch.sparse.sum(batch_holder.vars_const_graph, dim=-1).to_dense()
        vars_scaler = torch.unsqueeze(vars_scaler, dim=-1)

        for i in range(self.pass_steps):
            # TODO: Noise for queries
            query = self.make_query(variables)
            query = torch.sigmoid(query)

            left_side_value = torch.sparse.mm(batch_holder.vars_const_graph.t(), query)
            const_loss = torch.relu(left_side_value - const_values)
            const_loss1 = torch.relu(const_values - left_side_value)

            obj_loss = query * obj_multipliers
            const_gradient = torch.autograd.grad([const_loss.sum() + obj_loss.sum()], [query], retain_graph=True)[0]

            const_msg = torch.cat([constraints, const_loss / const_scaler, const_loss1 / const_scaler], dim=-1)
            const_tmp = self.constraint_update(const_msg)
            constraints = const_tmp[:, :self.feature_maps] + 0.5 * constraints

            const2var_msg = torch.sparse.mm(batch_holder.vars_const_graph, const_tmp[:, self.feature_maps:])

            var_msg = torch.cat([variables, const2var_msg / vars_scaler, obj_loss, const_gradient, obj_multipliers], dim=-1)
            variables = self.variable_update(var_msg) + 0.5 * variables

            out_vars = self.output(variables)
            int_noise = self.noise.sample(out_vars.size()).cuda()

            # Noise is not applied to variables that doesn't have integer constraint
            masked_int_noise = 0.5 * int_noise * torch.unsqueeze(batch_holder.integer_mask, dim=-1)

            if self.training:
                out_vars += masked_int_noise

            out = torch.sigmoid(out_vars)
            outputs.append(out)

            constraints = constraints.detach() * 0.2 + constraints * 0.8
            variables = variables.detach() * 0.2 + variables * 0.8

        return outputs, out_vars
