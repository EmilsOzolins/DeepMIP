from typing import List

import torch


class IPInstance:
    """
    Builds Integer Programming instance from individual constraints.
    """

    def __init__(self, variable_count=None) -> None:
        self._indices = []
        self._multipliers = []
        self._right_side_values = []
        self._current_constraint_index = 0
        self._max_var_index = variable_count - 1 if variable_count else 0

    def greater_or_equal(self, variable_indices: List[int],
                         variable_multipliers: List[float],
                         right_side_value: float
                         ) -> 'IPInstance':
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
                      ) -> 'IPInstance':
        """
        Adds less or equal constraint to the instance.
        Accepts constraint in the form a_0 * x_0 + a_1 * x_1 + ... + a_i * x_i <= b , where:
         * a values are variable_multipliers
         * variable_indices are vector of variable indices that are present in constraint
         * b is right_side_value.
        """
        self._max_var_index = max(max(variable_indices), self._max_var_index)

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
              ) -> 'IPInstance':
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

    @property
    def edge_indices(self):
        i = [x for x, _ in self._indices]
        j = [x for _, x in self._indices]
        return torch.as_tensor([i, j])

    @property
    def edge_values(self):
        return torch.as_tensor(self._multipliers, dtype=torch.float32)

    @property
    def constraints_values(self):
        return torch.as_tensor(self._right_side_values, dtype=torch.float32)

    @property
    def next_var_index(self):
        return self._max_var_index + 1

    @property
    def next_constraint_index(self):
        return self._current_constraint_index

    @property
    def size(self):
        return self.next_var_index, self.next_constraint_index
