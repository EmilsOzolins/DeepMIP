import torch
import torch.nn as nn


class MIPNetwork(torch.nn.Module):

    def __init__(self, output_bits, feature_maps=64, pass_steps=1):
        super().__init__()

        self.feature_maps = feature_maps
        self.pass_steps = pass_steps

        self.constraint_update = nn.Sequential(
            nn.Linear(self.feature_maps * 2, self.feature_maps * 2),
            nn.ReLU(),
            nn.Linear(self.feature_maps * 2, self.feature_maps)
        )

        self.variable_update = nn.Sequential(
            nn.Linear(self.feature_maps * 2, self.feature_maps * 2),
            nn.ReLU(),
            nn.Linear(self.feature_maps * 2, self.feature_maps)
        )

        self.output = nn.Sequential(
            nn.Linear(self.feature_maps, self.feature_maps * 2),
            nn.ReLU(),
            nn.Linear(self.feature_maps * 2, output_bits)
        )

        self.prepare_cond = nn.Sequential(
            nn.Linear(1, self.feature_maps * 2),
            nn.ReLU(),
            nn.Linear(self.feature_maps * 2, self.feature_maps)
        )

        self.step = 0

    def forward(self, adj_matrix: torch.sparse.Tensor, conditions_values: torch.Tensor):
        """
        :param adj_matrix: Adjacency matrix of MIP factor graph with size [var_count x constraint_count]
        :return: variable assignments with the size [var_count]
        """
        var_count, const_count = adj_matrix.size()

        variables = torch.ones([var_count, self.feature_maps]).cuda()
        constraints = self.prepare_cond(torch.unsqueeze(conditions_values, dim=-1))

        adj_matrix = adj_matrix.coalesce()
        adj_matrix = torch.sparse_coo_tensor(adj_matrix.indices(), torch.abs(adj_matrix.values()))

        # TODO: Use embedding for edges also?

        for i in range(self.pass_steps):
            var2const_msg = torch.mm(adj_matrix.t(), variables)
            var2const_msg = torch.cat([constraints, var2const_msg], dim=-1)
            constraints = self.constraint_update(var2const_msg)

            const2var_msg = torch.mm(adj_matrix, constraints)
            const2var_msg = torch.cat([variables, const2var_msg], dim=-1)
            variables = self.variable_update(const2var_msg)

        assignments = self.output(variables)

        # self.noise = torch.distributions.Normal(0, 16)
        assignments = torch.sigmoid(assignments)  # + self.noise.sample(assignments.size()).cuda())

        return assignments
