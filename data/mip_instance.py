from typing import List
from random import sample

import torch
from ortools.linear_solver import pywraplp


class MIPInstance:
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

        self._objective_multipliers = []
        self._objective_indices = []
        self._objective_set = False

        self._integer_indices = set()
        self._drop_percentage = 0.05
        self._fix_percentage = 0.05
        self._augment_steps = 10

    def greater_or_equal(self, variable_indices: List[int],
                         variable_multipliers: List[float],
                         right_side_value: float
                         ) -> 'MIPInstance':
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
                      ) -> 'MIPInstance':
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
              ) -> 'MIPInstance':
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
        self._objective_indices = variable_indices
        self._objective_multipliers = variable_multipliers
        return self

    def maximize_objective(self, variable_indices: List[int], variable_multipliers: List[float]):
        variable_multipliers = [-x for x in variable_multipliers]
        self.minimize_objective(variable_indices, variable_multipliers)
        return self

    def less(self):
        raise NotImplementedError()

    def greater(self):
        raise NotImplementedError()

    def integer_constraint(self, variable_indices: List[int]):
        """ Variables with this constraint will be integers. Rest of the variables will be floats.
        """
        self._integer_indices.update(variable_indices)
        return self

    def drop_random_constraints(self):
        # randomly select indices for constraints to drop
        n_dropped_constraints = int(self._current_constraint_index * self._drop_percentage)
        dropped_constraint_indices = sample(range(self._current_constraint_index), n_dropped_constraints)
        remaining_constraint_indices = [i for i in range(self._current_constraint_index) if i not in dropped_constraint_indices]

        # drop values from indices and multipliers corresponding to the dropped constraints
        for i in reversed(range(len(self._indices))):
            if self._indices[i][1] in dropped_constraint_indices:
                self._indices.pop(i)
                self._multipliers.pop(i)
        
        # renumber the constraints so that there are no gaps
        for i in range(len(self._indices)):
            old_constraint_index = self._indices[i][1]
            new_constraint_index = remaining_constraint_indices.index(self._indices[i][1])
            self._indices[i] = (self._indices[i][0], new_constraint_index)

        # drop values from right_side_values corresponding to the dropped constraints
        for i in reversed(range(len(self._right_side_values))):
            if i in dropped_constraint_indices:
                self._right_side_values.pop(i)

        # drop vars no longer existing from self._integer_indices
        variable_indices = sorted(list(set([pair[0] for pair in self._indices]))) # existing unique variable indices
        updated_integer_indices = self._integer_indices.copy()
        for i in self._integer_indices:
            if i not in variable_indices:
                updated_integer_indices.remove(i)
        self._integer_indices = updated_integer_indices

        # drop vars no longer existing from self._objective_indices and self._objective_multipliers
        for i in reversed(range(len(self._objective_indices))):
            if self._objective_indices[i] not in variable_indices:
                self._objective_indices.pop(i)
                self._objective_multipliers.pop(i)
                

        self._max_var_index = max(max(variable_indices), self._max_var_index)
        self._current_constraint_index -= n_dropped_constraints

        return self

    def fix_random_variables(self):
        # get existing unique variable indices
        variable_indices = sorted(list(set([pair[0] for pair in self._indices])))

        # randomly select indices for variables to fix
        n_fixed_variables = int(len(variable_indices) * self._fix_percentage)
        fixed_variable_indices = sample(variable_indices, n_fixed_variables)

        # solve the instance with OR-Tools and fix the selected variables
        solution = self.solve()
        for i in solution:
            if i in fixed_variable_indices:
                self.equal([i], [1], solution[i])

        return self

    def solve(self):
        solver = pywraplp.Solver.CreateSolver('SCIP')

        variable_indices = sorted(list(set([pair[0] for pair in self._indices]))) # existing unique variable indices
        int_variable_indices = sorted(list(self._integer_indices))
        num_variable_indices = [i for i in variable_indices if i not in int_variable_indices]

        int_variables = [solver.IntVar(0, solver.infinity(), str(i)) for i in int_variable_indices]
        num_variables = [solver.NumVar(0, solver.infinity(), str(i)) for i in num_variable_indices]
        variable_dict = dict(list(zip(int_variable_indices,int_variables)) + list(zip(num_variable_indices,num_variables)))

        # add each constraint to the solver
        for c in range(self._current_constraint_index):
            c_variable_indices_pairs = [pair for pair in self._indices if pair[1] == c]
            c_first_var_pair_index = self._indices.index(c_variable_indices_pairs[0])
            c_last_var_pair_index = self._indices.index(c_variable_indices_pairs[-1])
            c_variable_indices = [pair[0] for pair in c_variable_indices_pairs]
            c_weights = self._multipliers[c_first_var_pair_index:c_last_var_pair_index + 1]
            c_left_side = sum([c_weights[i] * variable_dict[c_variable_indices[i]] for i in range(len(c_variable_indices))])
            solver.Add(c_left_side <= self._right_side_values[c])

        # -1 multiplier because this class has minimize as default but ortools has maximize as default
        solver.Maximize(sum([variable_dict[self._objective_indices[i]] * self._objective_multipliers[i] * -1 for i in range(len(self._objective_indices))]))
        solver.Solve()

        solution_var_dict = {}
        for key in variable_dict:
            if key in int_variables:
                solution_var_dict[key] = round(variable_dict[key].solution_value())
            else:
                solution_var_dict[key] = variable_dict[key].solution_value()

        # print(solver.Objective().Value())
        return solution_var_dict
    
    def augment(self):
        for i in range(self._augment_steps):
            self.drop_random_constraints()
            self.fix_random_variables()
        return self

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

    @property
    def integer_variables(self):
        return torch.as_tensor(list(self._integer_indices), dtype=torch.int64)
