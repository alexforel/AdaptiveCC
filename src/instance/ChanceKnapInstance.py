import numpy as np

from src.instance.ChanceInstance import ChanceInstance


class ChanceKnapInstance(ChanceInstance):
    """Multi-dimensional knapsack instance."""

    def __init__(self, file_location, continuous_var, epsilon):
        super(ChanceKnapInstance, self).__init__(file_location)
        self.epsilon = epsilon
        self.continuous_var = continuous_var
        self.read_data()
        self.parse_data()

    #   - - - Private methods - - -
    def _parse_indices(self):
        """
        Parse indices relevant for the instance. Read:
        instance_info, nb_vars, nb_scenarios
        """
        self.instance_info = np.array(self.complete_file[0])
        self.nb_vars = int(self.instance_info[0])
        self.nb_scenarios = int(self.instance_info[2])

    def _parse_model_data(self):
        """
        Parse the CCLP model data. Read:
        vector_c, matrices_A, vectors_b, proba.
        """
        self.vector_c = np.array(self.complete_file[1])
        matrices_A_list = [
            [self.complete_file[2+s*self.nb_constraints[s]+i]
             for i in range(self.nb_constraints[s])]
            for s in range(self.nb_scenarios)]
        self.matrices_A = np.array(matrices_A_list)
        self.vectors_b = np.broadcast_to(
            self.complete_file[-1],
            (self.nb_scenarios, self.nb_constraints[0])).copy(order='C')
        self.proba = np.full(self.nb_scenarios, 1/self.nb_scenarios)
        # Rounding to avoid numerical errors
        self.matrices_A = np.round(self.matrices_A, decimals=1)
        self.vectors_b = np.round(self.vectors_b, decimals=1)
        self.vector_c = np.round(self.vector_c, decimals=1)

    def _parse_additional_parameters(self):
        """
        Parse the additional parameters necessary for the model. Read:
        var_type, nb_constraints, var_lb, var_ub.
        """
        self.var_type = np.full(self.nb_vars, self.continuous_var)
        self.nb_constraints = np.full(self.nb_scenarios,
                                      int(self.instance_info[1])).astype("int")
        if self.continuous_var:
            self.var_lb = np.full(self.nb_vars, 0)
            self.var_ub = np.full(self.nb_vars, 1)
        else:
            self.var_lb = np.full(self.nb_vars, -float('inf'))
            self.var_ub = np.full(self.nb_vars, float('inf'))

    @staticmethod
    def _get_constraint_violation(A_si, var_x, b_si):
        """
        Calculate the amount by which the variable var_x
        violates the single constraint:
                A_si * var_x <= b_si
        """
        return max(A_si.dot(var_x) - b_si, 0)

    @staticmethod
    def _is_scenario_infeasible(scenario_infeasibility):
        """
        A scenario is infeasible if at least one of
        its constraint is violated.
        """
        if len(scenario_infeasibility) > 0:
            return True
        else:
            return False

    def _get_scenario_infeasibility(self, var_x_val):
        """Measure the constraint violation for each scenario/constraint."""
        scenario_infeasibility = dict()
        # Iterate over every scenario/constraint and check for feasibility
        for s in range(self.nb_scenarios):
            scenario_infeasibility[s] = []
            for i in range(self.nb_constraints[s]):
                violation = self._get_constraint_violation(
                    self.matrices_A[s, i, :], var_x_val, self.vectors_b[s, i])
                TOLERANCE = 1e-6
                # Only add violation if larger than tolerance
                if violation > TOLERANCE:
                    rescaling = np.linalg.norm(self.matrices_A[s, i, :])
                    if rescaling != 0:
                        violation = violation/rescaling
                    scenario_infeasibility[s].append(violation)
        return scenario_infeasibility

    def _get_infeasible_scenarios(self, scenario_infeasibility):
        """Return boolean flag: True if scenario infeasible."""
        is_scenario_infeasible = []
        for s in range(self.nb_scenarios):
            is_scenario_infeasible.append(self._is_scenario_infeasible(
                scenario_infeasibility[s]))
        return is_scenario_infeasible

    #   - - - Public methods - - -
    def is_feasible(self, var_x_val):
        """
        Process the given point x_var. Determines if it is feasible
        for the CCLP, how many scenarios are satisfied and which one are
        feasible or not.
        """
        # Get constraint violation and collect infeasible scenarios
        scenario_infeasibility = self._get_scenario_infeasibility(var_x_val)
        is_scenario_infeasible = self._get_infeasible_scenarios(
            scenario_infeasibility)
        # Check feasiblity using amount of infeasible scenarios
        nb_infeasible_scenarios = np.sum(is_scenario_infeasible)
        nb_scen_tolerance = self.epsilon*self.nb_scenarios
        sol_is_feasible = nb_infeasible_scenarios <= nb_scen_tolerance
        return (sol_is_feasible, nb_infeasible_scenarios, nb_scen_tolerance,
                is_scenario_infeasible, scenario_infeasibility)

    def check_feasibility(self, var_x_val):
        """
        Checks the feasibility of a given solution,
        print summary of feasibility, and
        returns a boolean flag.
        """
        (sol_is_feasible, nb_infeasible_scenarios,
         nb_scen_tolerance, is_scenario_infeasible,
         scenario_infeasibility) = self.is_feasible(var_x_val)
        self.is_scenario_infeasible = is_scenario_infeasible
        self.infeasible_scenarios = [
            i for i, x in enumerate(is_scenario_infeasible) if x]

        # Get total violation of a scenario
        self.max_infeasibility_scenarios = np.zeros(self.nb_scenarios)
        for s in range(self.nb_scenarios):
            if len(scenario_infeasibility[s]) > 0:
                self.max_infeasibility_scenarios[s] = np.max(
                    scenario_infeasibility[s])

        # Print summary of feasibility
        print('\n Checking feasibility of upper bound solution:')
        print('  - Nb. infeasible scenarios: ', nb_infeasible_scenarios)
        print('  - Tolerance on nb infeas. scenarios: ', nb_scen_tolerance)
        print('  - Solution feasible: ', sol_is_feasible)
        print('  - Total constraint violations:',
              np.sum(self.max_infeasibility_scenarios))
        return sol_is_feasible

    def get_feasible_scenarios(self):
        """
        Getter for the feasibility of a scenario
        for the var_x_val provided in check_feasibility.

        Returns:
        _ (array of boolean): boolean array with True
                              for each scenario if it is
                              feasible for the previous solution
        """
        return self.max_infeasibility_scenarios == 0

    def get_sum_infeasibility(self):
        """
        Getter for the total infeasibility of a scenario
        for the var_x_val provided in check_feasibility.
        """
        return self.max_infeasibility_scenarios
