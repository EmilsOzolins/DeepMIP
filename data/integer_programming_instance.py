from typing import List

import torch


class IPInstanceBuilder:
    """
    Builds Integer Programming instance from individual constraints.
    """

    def __init__(self) -> None:
        self._indices = []
        self._multipliers = []
        self._right_side_values = []
        self._current_constraint_index = 0

    def greater_or_equal(self, variable_indices: List[int],
                         variable_multipliers: List[float],
                         right_side_value: float
                         ) -> 'IPInstanceBuilder':
        """
        Adds greater or equal constraint to the instance.
        Accepts constraint in the form a_0 * x_0 + a_1 * x_1 + ... + a_i * x_i >= b , where:
         * a values are variable_multipliers
         * variable_indices are vector of variable indices that are present in constraint
         * b is right_side_value.
        """

        variable_multipliers = [-x for x in variable_multipliers]  # change the direction of inequality
        right_side_value = -right_side_value

        self.less_or_equal(variable_indices, variable_multipliers, right_side_value)
        return self

    def less_or_equal(self, variable_indices: List[int],
                      variable_multipliers: List[float],
                      right_side_value: float
                      ) -> 'IPInstanceBuilder':
        """
        Adds less or equal constraint to the instance.
        Accepts constraint in the form a_0 * x_0 + a_1 * x_1 + ... + a_i * x_i <= b , where:
         * a values are variable_multipliers
         * variable_indices are vector of variable indices that are present in constraint
         * b is right_side_value.
        """
        for idx, a in zip(variable_indices, variable_multipliers):
            self._indices.append((idx, self._current_constraint_index))
            self._multipliers.append(a)

        self._right_side_values.append(right_side_value)

        # Increase the constraint index
        self._current_constraint_index += 1

        return self

    def equal(self, variable_indices: List[int],
              variable_multipliers: List[float],
              right_side_value: float
              ) -> 'IPInstanceBuilder':
        """
        Adds equal constraint to the instance.
        Accepts constraint in the form a_0 * x_0 + a_1 * x_1 + ... + a_i * x_i = b , where:
         * a values are variable_multipliers
         * variable_indices are vector of variable indices that are present in constraint
         * b is right_side_value.
        """
        self.less_or_equal(variable_indices, variable_multipliers, right_side_value)
        self.greater_or_equal(variable_indices, variable_multipliers, right_side_value)
        return self

    def create(self, device=torch.device('cuda:0')):
        """
        Encodes the instance as adjacency matrix of factor graph and right sight as scalar feature vector.
        Method returns PyTorch tensors placed on the device.
        """
        i = [x for x, _ in self._indices]
        j = [x for _, x in self._indices]

        adj_matrix = torch.sparse_coo_tensor(
            torch.tensor([i, j]),
            torch.tensor(self._multipliers),
            dtype=torch.float32,
            device=device
        )

        return adj_matrix, torch.tensor(self._right_side_values, dtype=torch.float32, device=device)
