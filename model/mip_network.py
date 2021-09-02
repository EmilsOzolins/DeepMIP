import torch
import torch.nn as nn

from model.normalization import PairNorm


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

        self.variable_update = nn.Sequential(
            nn.Linear(self.feature_maps * 3, self.feature_maps),
            nn.ReLU(),
            nn.Linear(self.feature_maps, self.feature_maps),
            PairNorm()
        )

        self.objective_update = nn.Sequential(
            nn.Linear(self.feature_maps * 2, self.feature_maps),
            nn.ReLU(),
            nn.Linear(self.feature_maps, self.feature_maps),
            PairNorm()
        )

        self.output = nn.Sequential(
            nn.Linear(self.feature_maps, self.feature_maps),
            nn.ReLU(),
            nn.Linear(self.feature_maps, output_bits)
        )

        self.prepare_cond = nn.Sequential(
            nn.Linear(1, self.feature_maps),
            nn.ReLU(),
            nn.Linear(self.feature_maps, self.feature_maps),
            PairNorm()
        )

        self.noise = torch.distributions.Normal(0, 1)

        self.step = 0
        self.powers_of_two = torch.as_tensor([2 ** k for k in range(0, output_bits)], dtype=torch.float32, device=torch.device('cuda:0'))

    def forward(self, adj_matrix: torch.sparse.Tensor, conditions_values: torch.Tensor,
                vars_obj_graph: torch.sparse.Tensor, const_inst_graph: torch.sparse.Tensor):
        """
        :param adj_matrix: Adjacency matrix of MIP factor graph with size [var_count x constraint_count]
        :return: variable assignments with the size [var_count]
        """
        var_count, const_count = adj_matrix.size()
        _, objective_count = vars_obj_graph.size()

        variables = torch.ones([var_count, self.feature_maps], device=torch.device('cuda:0'))
        constraints = self.prepare_cond(torch.unsqueeze(conditions_values, dim=-1))
        objectives = torch.ones([objective_count, self.feature_maps], device=torch.device('cuda:0'))

        binary_outputs = []
        decimal_outputs = []

        for i in range(self.pass_steps):
            var2obj_msg = torch.sparse.mm(vars_obj_graph.t(), variables)
            var2obj_msg = torch.cat([objectives, var2obj_msg], dim=-1)
            objectives = self.objective_update(var2obj_msg)

            var2const_msg = torch.sparse.mm(adj_matrix.t(), variables)
            const_msg = torch.cat([constraints, var2const_msg], dim=-1)
            constraints = self.constraint_update(const_msg)

            const2var_msg = torch.sparse.mm(adj_matrix, constraints)
            obj2var_msg = torch.sparse.mm(vars_obj_graph, objectives)
            var_msg = torch.cat([variables, const2var_msg, obj2var_msg], dim=-1)
            variables = self.variable_update(var_msg)

            out_vars = self.output(variables)
            out = torch.sigmoid(out_vars + self.noise.sample(out_vars.size()).cuda())

            binary_outputs.append(out)

            decimal_pred = torch.sum(self.powers_of_two * out, dim=-1, keepdim=True)
            decimal_outputs.append(decimal_pred)

            constraints = constraints.detach() * 0.2 + constraints * 0.8
            variables = variables.detach() * 0.2 + variables * 0.8
            objectives = objectives.detach() * 0.2 + objectives * 0.8

        return binary_outputs, decimal_outputs
