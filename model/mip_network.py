import torch
import torch.nn as nn


class MIPNetwork(torch.nn.Module):

    def __init__(self, output_bits, feature_maps=32, pass_steps=3):
        super().__init__()

        self.feature_maps = feature_maps
        self.pass_steps = pass_steps

        self.constraint_update = nn.Sequential(
            nn.Linear(self.feature_maps * 3, self.feature_maps * 2),
            nn.ReLU(),
            nn.Linear(self.feature_maps * 2, self.feature_maps),
            nn.LayerNorm(self.feature_maps, elementwise_affine=False)
        )

        self.variable_update = nn.Sequential(
            nn.Linear(self.feature_maps * 2, self.feature_maps * 2),
            nn.ReLU(),
            nn.Linear(self.feature_maps * 2, self.feature_maps),
            nn.LayerNorm(self.feature_maps, elementwise_affine=False)
        )

        self.output = nn.Sequential(
            nn.Linear(self.feature_maps, self.feature_maps * 2),
            nn.ReLU(),
            nn.Linear(self.feature_maps * 2, output_bits)
        )

        self.prepare_cond = nn.Sequential(
            nn.Linear(1, self.feature_maps * 2),
            nn.ReLU(),
            nn.Linear(self.feature_maps * 2, self.feature_maps),
            nn.LayerNorm(self.feature_maps, elementwise_affine=False)
        )

        self.noise = torch.distributions.Normal(0, 1)

        self.step = 0

    def forward(self, adj_matrix: torch.sparse.Tensor, conditions_values: torch.Tensor):
        """
        :param adj_matrix: Adjacency matrix of MIP factor graph with size [var_count x constraint_count]
        :return: variable assignments with the size [var_count]
        """
        var_count, const_count = adj_matrix.size()

        variables = torch.ones([var_count, self.feature_maps], device=torch.device('cuda:0'))
        constraints = emb_value = self.prepare_cond(torch.unsqueeze(conditions_values, dim=-1))

        for i in range(self.pass_steps):
            var2const_msg = torch.mm(adj_matrix.t(), variables)
            var2const_msg = torch.cat([constraints, emb_value, var2const_msg], dim=-1)
            constraints = self.constraint_update(var2const_msg)

            const2var_msg = torch.mm(adj_matrix, constraints)
            const2var_msg = torch.cat([variables, const2var_msg], dim=-1)
            variables = self.variable_update(const2var_msg)

        out_vars = self.output(variables)
        return torch.sigmoid(out_vars)  # + self.noise.sample(out_vars.size()).cuda())
