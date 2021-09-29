import torch
import torch.nn as nn
from model.normalization import NodeNorm, PairNorm
from utils.data_utils import sparse_func, InputDataHolder

def sample_triangular(shape):
    sample1 = torch.rand(shape).cuda()
    sample2 = torch.rand(shape).cuda()
    return sample1 + sample2


class MIPNetwork(torch.nn.Module):

    def __init__(self, output_bits, feature_maps=64, pass_steps=3, summary=None):
        super().__init__()

        self.feature_maps = feature_maps
        self.pass_steps = pass_steps
        self.summary = summary
        self.global_step = 0

        self.constraint_update = nn.Sequential(
            nn.Linear(self.feature_maps * 2, self.feature_maps),
            PairNorm(subtract_mean=False),
            nn.ReLU(),
            nn.Linear(self.feature_maps, self.feature_maps * 2),
        )

        self.make_query = nn.Sequential(
            nn.Linear(self.feature_maps + 4, self.feature_maps),
            nn.ReLU(),
            nn.Linear(self.feature_maps, self.feature_maps),
        )

        self.variable_update = nn.Sequential(
            nn.Linear(self.feature_maps * 3 + 1, self.feature_maps),
            PairNorm(subtract_mean=False),
            nn.ReLU(),
            nn.Linear(self.feature_maps, self.feature_maps),
        )

        self.output = nn.Sequential(
            nn.Linear(self.feature_maps, self.feature_maps),
            nn.ReLU(),
            nn.Linear(self.feature_maps, output_bits)
        )

        self.noise = torch.distributions.Normal(0, 1)

        self.step = 0

    def forward(self, batch_holder: InputDataHolder, device):
        # TODO: Experiment with disentangled architecture
        var_count, const_count = batch_holder.vars_const_graph.size()
        _, objective_count = batch_holder.vars_inst_graph.size()

        variables = torch.ones([var_count, self.feature_maps], device=device)
        constraints = torch.ones([const_count, self.feature_maps], device=device)

        outputs = []

        const_values = torch.unsqueeze(batch_holder.const_values, dim=-1)
        obj_multipliers = torch.unsqueeze(batch_holder.objective_multipliers, dim=-1)
        obj_multipliers /= torch.mean(torch.abs(obj_multipliers)) + 1e-6

        abs_graph = sparse_func(batch_holder.vars_const_graph, torch.abs)
        unit_graph = sparse_func(batch_holder.vars_const_graph, torch.sign)
        # unit_graph = make_sparse_unit(batch_holder.vars_const_graph)
        # const_scaler = torch.sqrt(torch.sparse.sum(abs_graph, dim=0).to_dense()) + 1e-6
        # denom = torch.sparse.sum(unit_graph, dim=0).to_dense()+1e-6
        # const_scaler = torch.unsqueeze(const_scaler/denom, dim=-1)

        # TODO: Experiment with mean
        const_scaler = torch.sparse.sum(abs_graph, dim=0).to_dense() + 1e-6
        const_scaler = torch.unsqueeze(const_scaler, dim=-1)

        # TODO: Experiment with mean
        vars_scaler = torch.sparse.sum(abs_graph, dim=-1).to_dense() + 1e-6
        vars_scaler = torch.unsqueeze(vars_scaler, dim=-1)

        int_mask = torch.unsqueeze(batch_holder.integer_mask, dim=-1)

        for i in range(self.pass_steps):
            with torch.enable_grad():
                var_noisy = torch.cat([variables, self.noise.sample([var_count, 4]).cuda()], dim=-1)
                query = self.make_query(var_noisy)
                query = torch.sigmoid(query) * int_mask + query * (1 - int_mask)

                left_side_value = torch.sparse.mm(batch_holder.vars_const_graph.t(), query)
                left_side_value = (left_side_value - const_values) / const_scaler
                const_loss = left_side_value
                #const_loss1 = torch.relu(left_side_value)

                # obj_loss = query * obj_multipliers
                const_gradient = torch.autograd.grad([const_loss.sum()], [query], retain_graph=True)[0]
                # const_gradient1 = torch.autograd.grad([const_loss1.sum()], [query], retain_graph=True)[0]

            const_msg = torch.cat([constraints, const_loss], dim=-1)
            const_tmp = self.constraint_update(const_msg)
            constraints = const_tmp[:, :self.feature_maps] + 0.5 * constraints

            const2var_msg = torch.sparse.mm(unit_graph, const_tmp[:, self.feature_maps:]) / vars_scaler

            var_msg = torch.cat([variables, const2var_msg, const_gradient, obj_multipliers], dim=-1)
            variables = self.variable_update(var_msg) + 0.5 * variables

            out_vars = self.output(variables)
            int_noise = self.noise.sample(out_vars.size()).cuda()
            #int_noise = sample_triangular(out_vars.size())

            # Noise is not applied to variables that doesn't have integer constraint
            if self.training:
                out_vars += 1.5 * int_noise * int_mask

            out = torch.sigmoid(out_vars) * int_mask + out_vars * (1 - int_mask)
            outputs.append(out)

            constraints = constraints.detach() * 0.2 + constraints * 0.8
            variables = variables.detach() * 0.2 + variables * 0.8

        # self.summary.add_histogram("query", query, self.global_step)
        # self.summary.add_histogram("query_constr", left_side_value, self.global_step)
        # self.summary.add_histogram("query_grad", const_gradient, self.global_step)
        # #self.summary.add_histogram("obj_loss", obj_loss, self.global_step)
        # self.summary.add_histogram("obj_multipliers", obj_multipliers, self.global_step)
        # self.summary.add_histogram("const2var_msg", const2var_msg, self.global_step)
        # self.summary.add_histogram("variables_data", variables, self.global_step)
        # self.summary.add_histogram("constraints_data", constraints, self.global_step)
        # self.summary.add_histogram("const_scaler", const_scaler, self.global_step)
        # self.summary.add_histogram("vars_scaler", vars_scaler, self.global_step)

        return outputs, out_vars
