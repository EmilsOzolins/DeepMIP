import torch
import torch.nn as nn


class MIPNetwork(torch.nn.Module):

    def __init__(self, output_bits, feature_maps=16, pass_steps=2):
        super().__init__()

        self.feature_maps = feature_maps
        self.pass_steps = pass_steps

        self.constraint_update = nn.Linear(self.feature_maps * 2, self.feature_maps)
        self.variable_update = nn.Linear(self.feature_maps * 2, self.feature_maps)
        self.output = nn.Linear(self.feature_maps, self.feature_maps)
        self.output2 = nn.Linear(self.feature_maps, output_bits)

        self.noise = torch.distributions.Normal(0, 8)

    def forward(self, adj_matrix: torch.sparse.Tensor, conditions_values: torch.Tensor):
        """
        :param adj_matrix: Adjacency matrix of MIP factor graph with size [var_count x constraint_count]
        :return: variable assignments with the size [var_count]
        """

        var_count, const_count = adj_matrix.size()

        variables = torch.ones([var_count, self.feature_maps])
        # TODO: Embed conditions values into the constraints
        constraints = torch.zeros([const_count, self.feature_maps])

        for i in range(self.pass_steps):
            var2const_msg = torch.sparse.mm(adj_matrix.t(), variables)
            var2const_msg = torch.cat([constraints, var2const_msg], dim=-1)
            constraints = self.constraint_update(var2const_msg)
            constraints = torch.relu(constraints)

            const2var_msg = torch.sparse.mm(adj_matrix, constraints)
            const2var_msg = torch.cat([variables, const2var_msg], dim=-1)
            variables = self.variable_update(const2var_msg)
            variables = torch.relu(variables)

        assignments = self.output(variables)
        assignments = torch.relu(assignments)
        assignments = self.output2(assignments)
        assignments = torch.sigmoid(assignments + self.noise.sample(assignments.size()))

        return assignments
