from typing import List

import torch


class IPInstance:
    """
    Builds Integer Programming instance from individual constraints.
    This is abstraction over lowe-level formulation to make formulating arbitrary problems as MIP easy.
    """

    def __init__(self, variable_count=None) -> None:
        self._indices = []
        self._multipliers = []
        self._right_side_values = []
        self._current_constraint_index = 0
        self._max_var_index = variable_count - 1 if variable_count else 0

        self._objective_indices = []
        self._objective_multipliers = []

        self._objective_set = False

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

    def minimize_objective(self, variable_indices: List[int], variable_multipliers: List[float]):
        """
        This objective function will be minimized
        """
        if self._objective_set:
            raise ValueError("Objective already set, can't set it second time!")

        self._objective_set = True
        self._objective_indices += variable_indices
        self._objective_multipliers += variable_multipliers

    def maximize_objective(self, variable_indices: List[int], variable_multipliers: List[float]):
        variable_multipliers = [-x for x in variable_multipliers]
        self.minimize_objective(variable_indices, variable_multipliers)

    def less(self):
        raise NotImplementedError()

    def greater(self):
        raise NotImplementedError()

    @property
    def variables_constraints_graph(self):
        i = [x for x, _ in self._indices]
        j = [x for _, x in self._indices]
        return torch.as_tensor([i, j])

    @property
    def variables_constraints_values(self):
        return torch.as_tensor(self._multipliers, dtype=torch.float32)

    @property
    def right_values_of_constraints(self):
        return torch.as_tensor(self._right_side_values, dtype=torch.float32)

    @property
    def next_variable_index(self):
        return self._max_var_index + 1

    @property
    def next_constraint_index(self):
        return self._current_constraint_index

    @property
    def variables_constraints_graph_size(self):
        return self.next_variable_index, self.next_constraint_index

    @property
    def variables_objective_graph(self):
        graph_indices = [0] * len(self._objective_indices)
        return torch.as_tensor([self._objective_indices, graph_indices], dtype=torch.int32)

    @property
    def variables_objective_graph_values(self):
        return torch.as_tensor(self._objective_multipliers, dtype=torch.float32)

    @property
    def constraints_instance_graph(self):
        """ Returns adjacency matrix indices for clauses-instance graph
        """
        graph_indices = [0] * self.next_constraint_index
        clauses_indices = [i for i in range(self.next_constraint_index)]
        return torch.as_tensor([clauses_indices, graph_indices])

    @property
    def constraints_instance_graph_values(self):
        return torch.as_tensor([1] * self.next_constraint_index, dtype=torch.float32)

    @property
    def variables_instance_graph(self):
        """ Returns adjacency matrix indices for variables-instance graph
        """
        graph_indices = [0] * self.next_variable_index
        vars_indices = [i for i in range(self.next_variable_index)]
        return torch.as_tensor([vars_indices, graph_indices])

    @property
    def variables_instance_graph_values(self):
        return torch.as_tensor([1] * self.next_variable_index, dtype=torch.float32)
