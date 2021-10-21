import numpy as np
import torch
import torch.nn as nn

import hyperparams as params
from model.normalization import PairNorm
from utils.data_utils import sparse_func, InputDataHolder, sparse_dense_mul_1d


def sample_triangular(shape):
    sample1 = torch.rand(shape).cuda()
    sample2 = torch.rand(shape).cuda()
    return sample1 + sample2


def sample_logistic(shape, eps=1e-5):
    minval = eps
    maxval = 1 - eps
    sample = (minval - maxval) * torch.rand(shape).cuda() + maxval
    return torch.log(sample / (1 - sample)).detach()


class MIPNetwork(torch.nn.Module):

    def __init__(self, output_bits, feature_maps=64, pass_steps=3, summary=None):
        super().__init__()

        self.feature_maps = feature_maps
        self.pass_steps = pass_steps
        self.summary = summary
        self.global_step = 0
        self.use_preconditioning = False
        self.continuous_var_scale = 0.1

        self.constraint_update = nn.Sequential(
            nn.Linear(self.feature_maps * 2, self.feature_maps),
            PairNorm(subtract_mean=False),
            nn.ReLU(),
            nn.Linear(self.feature_maps, self.feature_maps * 2),
        )

        self.eq_constraint_update = nn.Sequential(
            nn.Linear(self.feature_maps * 2, self.feature_maps),
            PairNorm(subtract_mean=False),
            nn.ReLU(),
            nn.Linear(self.feature_maps, self.feature_maps * 2),
        )

        self.make_query = nn.Sequential(
            nn.Linear(self.feature_maps + 4, self.feature_maps),
            PairNorm(subtract_mean=False),
            nn.ReLU(),
            nn.Linear(self.feature_maps, self.feature_maps),
        )

        self.variable_update = nn.Sequential(
            nn.Linear(self.feature_maps * 4 + 2, self.feature_maps),
            PairNorm(subtract_mean=False),
            nn.ReLU(),
            nn.Linear(self.feature_maps, self.feature_maps),
        )

        self.output = nn.Sequential(
            nn.Linear(self.feature_maps, self.feature_maps),
            PairNorm(subtract_mean=False),
            nn.ReLU(),
            nn.Linear(self.feature_maps, output_bits)
        )

        self.noise = torch.distributions.Normal(0, 1)

        self.step = 0

    def forward(self, batch_holder: InputDataHolder, device):
        # TODO: Experiment with disentangled architecture
        var_count, const_count = batch_holder.vars_const_graph.size()
        _, eq_const_count = batch_holder.vars_eq_const_graph.size()
        _, objective_count = batch_holder.vars_inst_graph.size()

        const_values = torch.unsqueeze(batch_holder.const_values, dim=-1)
        eq_const_values = torch.unsqueeze(batch_holder.eq_const_values, dim=-1)
        relaxed_solution = torch.unsqueeze(batch_holder.relaxed_solution, dim=-1)

        variables = torch.ones([var_count, self.feature_maps], device=device) * relaxed_solution
        constraints = torch.sparse.mm(batch_holder.vars_const_graph.t(), variables)
        eq_constraints = torch.sparse.mm(batch_holder.vars_eq_const_graph.t(), variables)

        outputs = []

        obj_multipliers = torch.unsqueeze(batch_holder.objective_multipliers, dim=-1)
        obj_multipliers /= torch.sqrt(torch.mean(torch.square(obj_multipliers))) + 1e-6

        abs_graph = sparse_func(batch_holder.vars_const_graph, torch.square)
        # unit_graph_pos = sparse_func(batch_holder.vars_const_graph, lambda x: torch.greater(x, 0.).float())
        # unit_graph_neg = sparse_func(batch_holder.vars_const_graph, lambda x: torch.less(x, 0.).float())
        unit_graph_pos = sparse_func(batch_holder.vars_const_graph, torch.relu)
        unit_graph_neg = sparse_func(batch_holder.vars_const_graph, lambda x: torch.relu(-x))

        # unit_graph = make_sparse_unit(batch_holder.vars_const_graph)
        # const_scaler = torch.sqrt(torch.sparse.sum(abs_graph, dim=0).to_dense()) + 1e-6
        # denom = torch.sparse.sum(unit_graph, dim=0).to_dense()+1e-6
        # const_scaler = torch.unsqueeze(const_scaler/denom, dim=-1)

        # TODO: Experiment with mean
        const_scaler = torch.sparse.sum(abs_graph, dim=0).to_dense() + 1e-6
        const_scaler_1d = torch.sqrt(const_scaler)
        const_scaler = torch.unsqueeze(const_scaler_1d, dim=-1)

        if batch_holder.vars_eq_const_graph._nnz()>0:
            abs_graph_eq = sparse_func(batch_holder.vars_eq_const_graph, torch.square)
            eq_const_scaler = torch.sparse.sum(abs_graph_eq, dim=0).to_dense() + 1e-6
            eq_const_scaler_1d = torch.sqrt(eq_const_scaler)
            eq_const_scaler = torch.unsqueeze(eq_const_scaler_1d, dim=-1)
        else:
            eq_const_scaler = 1

        # TODO: Experiment with mean
        vars_scaler = torch.sparse.sum(abs_graph, dim=-1).to_dense() + 1e-6
        vars_scaler = torch.unsqueeze(torch.sqrt(vars_scaler), dim=-1)

        if self.use_preconditioning:
            abs_graph1 = sparse_dense_mul_1d(abs_graph, 1.0 / const_scaler_1d, dim=1)
            vars_regul = torch.sparse.sum(abs_graph1, dim=-1).to_dense() + 1e-6
            vars_regul = torch.unsqueeze(torch.sqrt(vars_regul), dim=-1)
            # divs = obj_multipliers*np.sqrt(var_count)/vars_regul
            logit_scalers = 1 / (vars_regul + obj_multipliers * params.objective_loss_scale * np.sqrt(var_count))
            logit_scalers = logit_scalers / torch.mean(logit_scalers)

        int_mask = torch.unsqueeze(batch_holder.integer_mask, dim=-1)

        for i in range(self.pass_steps):
            with torch.enable_grad():
                var_noisy = torch.cat([variables, self.noise.sample([var_count, 4]).cuda()], dim=-1)
                query = self.make_query(var_noisy)
                query = (query + 0.5) * int_mask + query * (1 - int_mask)

                left_side_value = torch.sparse.mm(batch_holder.vars_const_graph.t(), query)
                left_side_value = (left_side_value - const_values) / const_scaler
                const_loss = left_side_value
                # const_loss1 = torch.relu(left_side_value)

                # obj_loss = query * obj_multipliers
                # const_gradient = torch.autograd.grad([const_loss1.sum()], [query], retain_graph=True)[0]
                # const_gradient1 = torch.autograd.grad([const_loss1.sum()], [query], retain_graph=True)[0]

            const_msg = torch.cat([constraints, const_loss], dim=-1)
            const_tmp = self.constraint_update(const_msg)
            constraints = const_tmp[:, :self.feature_maps] + 0.5 * constraints  # TODO: No residual connections?

            constr_features = const_tmp[:, self.feature_maps:]
            const2var_msg_pos = torch.sparse.mm(unit_graph_pos, constr_features) / vars_scaler
            const2var_msg_neg = torch.sparse.mm(unit_graph_neg, constr_features) / vars_scaler

            eq_lhs = torch.sparse.mm(batch_holder.vars_eq_const_graph.t(), query)
            eq_loss = torch.square((eq_const_values - eq_lhs) / eq_const_scaler)
            eq_const_msg = torch.cat([eq_constraints, eq_loss], dim=-1)
            eq_const_tmp = self.eq_constraint_update(eq_const_msg)

            eq_constraints = eq_const_tmp[:, :self.feature_maps] + 0.5 * eq_constraints
            eq_const2var_msg = torch.sparse.mm(batch_holder.vars_eq_const_graph, eq_const_tmp[:, self.feature_maps:])

            var_msg = torch.cat([variables, const2var_msg_pos, const2var_msg_neg, eq_const2var_msg, obj_multipliers, int_mask], dim=-1)
            variables = self.variable_update(var_msg) + 0.5 * variables

            out_vars = self.output(variables)
            if self.use_preconditioning:
                out_vars = out_vars * logit_scalers + out_vars.detach() * (1 - logit_scalers)
            #int_noise = self.noise.sample(out_vars.size()).cuda()
            int_noise = sample_logistic(out_vars.size())
            #int_noise = torch.sign(out_vars)*torch.abs(int_noise)
            #mul_noise = torch.exp(self.noise.sample([1]).cuda()*0.2)
            #mul_noise = torch.abs(self.noise.sample(out_vars.size()).cuda() * 0.5)

            # Noise is not applied to variables that doesn't have integer constraint
            if self.training:
                #out_vars = (out_vars*mul_noise + 1*int_noise) * int_mask + out_vars*(1-int_mask)
                out_vars = out_vars + 1 * int_noise * int_mask

            out = torch.sigmoid(out_vars) * int_mask + out_vars * (1 - int_mask) * self.continuous_var_scale
            outputs.append(out)

            # constraints = constraints.detach() * 0.2 + constraints * 0.8
            # variables = variables.detach() * 0.2 + variables * 0.8

        if self.global_step % 100 == 0:
            # self.summary.add_histogram("query", query[:,0:4], self.global_step)
            # self.summary.add_histogram("query_constr", left_side_value[:,0:4], self.global_step)
            # self.summary.add_histogram("query_grad", const_gradient, self.global_step)
            # #self.summary.add_histogram("obj_loss", obj_loss, self.global_step)
            # self.summary.add_histogram("obj_multipliers", obj_multipliers, self.global_step)
            # self.summary.add_histogram("const2var_msg", const2var_msg_pos, self.global_step)
            # self.summary.add_histogram("variables_data", variables, self.global_step)
            # self.summary.add_histogram("constraints_data", constraints, self.global_step)
            # self.summary.add_histogram("const_scaler", const_scaler, self.global_step)
            # self.summary.add_histogram("vars_scaler", vars_scaler, self.global_step)
            #self.summary.add_histogram("mul_noise", mul_noise, self.global_step)
            if self.use_preconditioning: self.summary.add_histogram("logit_scaler", logit_scalers, self.global_step)
            # self.summary.add_histogram("divs", divs, self.global_step)

        return outputs, out_vars
