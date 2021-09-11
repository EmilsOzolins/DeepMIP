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
            nn.Linear(self.feature_maps, self.feature_maps),
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
            nn.Linear(self.feature_maps * 5, self.feature_maps),
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

        for i in range(self.pass_steps):
            const_query = self.make_query_constraints(variables)
            sig_const_query = torch.sigmoid(const_query)
            left_side_value = torch.sparse.mm(batch_holder.vars_const_graph.t(), sig_const_query)
            const_loss = torch.relu(left_side_value - const_values)
            const_loss1 = torch.relu(const_values-left_side_value)

            const_gradient = torch.autograd.grad([const_loss.sum()], [sig_const_query], retain_graph=True)[0]
            # var2const_msg = torch.sparse.mm(batch_holder.vars_const_graph.t(), variables)

            scalers = torch.sparse.sum(batch_holder.vars_const_graph, dim=0).to_dense()
            scalers = torch.unsqueeze(scalers, dim=-1)

            const_msg = torch.cat([constraints, const_loss / scalers, const_loss1/scalers], dim=-1)
            constraints = self.constraint_update(const_msg) + 0.5 * constraints

            obj_query = self.make_query_objective(variables)
            obj_query = torch.sigmoid(obj_query)
            obj_loss =  obj_query * obj_multipliers
            # obj_loss = self.combined_loss(torch.sigmoid(obj_query), batch_holder)

            obj_gradient = torch.autograd.grad([obj_loss.sum()], [obj_query], retain_graph=True)[0]# = obj_multipliers?

            scalers = torch.sparse.sum(batch_holder.vars_const_graph, dim=-1).to_dense()
            scalers = torch.unsqueeze(scalers, dim=-1)

            const2var_msg = torch.sparse.mm(batch_holder.vars_const_graph, constraints)
            var_msg = torch.cat([variables, const2var_msg / scalers, obj_loss, const_gradient, obj_gradient], dim=-1)
            variables = self.variable_update(var_msg) + 0.5 * variables

            out_vars = self.output(variables)
            int_noise = self.noise.sample(out_vars.size()).cuda()

            # Noise is not applied to variables that doesn't have integer constraint
            masked_int_noise = 0.5*int_noise * torch.unsqueeze(batch_holder.integer_mask, dim=-1)

            if self.training: out_vars += masked_int_noise
            out = torch.sigmoid(out_vars)
            outputs.append(out)

            constraints = constraints.detach() * 0.2 + constraints * 0.8
            variables = variables.detach() * 0.2 + variables * 0.8

        return outputs, out_vars

    def combined_loss(self, asn, batch_holder):
        """
        Makes objective loss dependent on constraint loss.
        """
        left_side = torch.sparse.mm(batch_holder.vars_const_graph.t(), asn)
        vars_in_const = torch.sparse.sum(batch_holder.binary_vars_const_graph, dim=0).to_dense()
        vars_in_const = torch.unsqueeze(vars_in_const, dim=-1)

        loss_c = torch.relu(left_side - torch.unsqueeze(batch_holder.const_values, dim=-1))
        loss_c /= vars_in_const

        loss_per_var = torch.sparse.mm(batch_holder.binary_vars_const_graph, loss_c)

        # TODO: Maybe zero is too strict and small leak should be allowed?
        mask = torch.isclose(loss_per_var, torch.zeros_like(loss_per_var)).float()
        mask = mask * 2 - 1
        loss_per_var = mask * (loss_per_var + 1)

        # TODO: If no objective, optimize constraint loss directly
        obj_multipliers = torch.unsqueeze(batch_holder.objective_multipliers, dim=-1)

        return obj_multipliers * loss_per_var * asn
