import numpy as np
from copy import copy
from gurobipy import GRB

from src.optim.DeterModel import DeterModel


class LowerBounder():
    """
    Provide lower bound to partitioned CCLP.
    """
    def __init__(self, chance_instance,
                 projection_method="rescaled_max_violation",
                 gap=1e-4, use_one_thread=True, verbose=False):
        self.chance_instance = chance_instance
        self.projection_method = projection_method
        self.nb_scenarios = self.chance_instance.get_nb_scenarios()
        self.feasibility_counter = np.zeros(self.nb_scenarios)
        # Initialize deterministic lower bound problem
        self.scenarios = [0]  # dummy scenario
        self.previous_vLB = -1e6
        self.deter_model = DeterModel(chance_instance)
        self.deter_model.build(self.scenarios, verbose=verbose)
        self.deter_model.grb_model.setParam(GRB.Param.MIPGap, gap)
        if use_one_thread:
            self.deter_model.grb_model.setParam(GRB.Param.Threads, 1)

    #   - - - Private methods - - -
    def _get_scenarios_violation_index(self):
        """Computes the rescales constraint violation of each scenario.

        Returns:
           scenario_violation (list[float]): the violation for every scenario
        """
        scenario_violation = np.zeros(self.nb_scenarios)
        for s in range(self.nb_scenarios):
            # Compute violation
            nb_constraints = self.chance_instance.get_nb_constraints(s)
            A = self.chance_instance.get_matrix_A(s)
            b = self.chance_instance.get_vector_b(s)
            for i in range(nb_constraints):
                violation = max(A[i, :].dot(self.xUB) - b[i], 0)
                # Rescale violation
                TOLERANCE = 1e-3
                if violation < TOLERANCE:
                    violation = 0
                rescaling = np.linalg.norm(A[i, :])
                if rescaling != 0:
                    violation = violation/rescaling
                # Find maximum violation over all scenarios
                scenario_violation[s] = max(
                    scenario_violation[s], violation)
        return scenario_violation

    def _get_index_scenarios(self):
        """
        Returns the index of all the scenarios based
        on the selected projection method.

        Returns:
            index_scenarios (list[int]): list of selected scenarios

        """
        if self.projection_method == "counter":
            index_scenarios = - self.feasibility_counter
        elif self.projection_method == "rescaled_max_violation":
            index_scenarios = self._get_scenarios_violation_index()
        return index_scenarios

    def _get_selected_scenarios(self, index_scenarios):
        """Returns the scenarios that have the largest index
           until the threshold of feasibility is satisfied.

        Returns:
            selected scenarios (list[int])
        """
        proba = self.chance_instance.get_proba()
        epsilon = self.chance_instance.get_epsilon()
        # Sort index_scenarios in increasing order
        if index_scenarios.shape[0] < self.nb_scenarios:
            raise ValueError("index_scenarios does not have the right size")
        sorted_scenarios = np.argsort(index_scenarios)
        # Add sorted scenarios to selected_scenarios set until
        # cumulative probability satisfies 1 - epsilon threshold
        index_counter = 0
        total_proba = 0
        TOL = 1e-7
        for s in sorted_scenarios:
            index_counter += 1
            scenario_proba = proba[s]
            total_proba += scenario_proba
            if total_proba >= (1 - epsilon - TOL):
                selected_scenarios = sorted_scenarios[0:index_counter]
                return selected_scenarios.tolist()

    def _find_best_incumbent(self, incumbents, vUB, vLB):
        """
        Find the best incumbent solution that is feasible to the original
        problem and has a better objective than the current vUB.
        """
        # Calculate the objective of all incumbent solutions
        vector_c = self.chance_instance.get_vector_c()
        objList = np.matmul(np.array(incumbents), vector_c)
        # Sort objective in decreasing order
        decreasing_indices = np.argsort(-objList)
        # Find which incumbents improved vLB while being lower than vUB
        TOL = 1e-8
        isImproving = (objList <= vUB) * (objList >= (vLB + TOL))
        for i in decreasing_indices:
            if isImproving[i]:
                obj = objList[i]
                x = incumbents[i]
                isFeasible = self.chance_instance.is_feasible(x)[0]
                if isFeasible:
                    # Found a new, improving feasible solution:
                    #    No need to continue since we sorted the
                    #    objectives in decreasing order
                    return True, x, obj
        return False, None, None

    def _solve_deter_model(self, scenarios):
        # Remove constraints that are not used anymore
        unused_scenarios = [s for s in self.scenarios if s not in scenarios]
        self.deter_model.remove(unused_scenarios)
        # Add constraints of new scenarios
        new_scenarios = [s for s in scenarios if s not in self.scenarios]
        self.deter_model.add(new_scenarios)
        # Store scenarios used in this iteration
        self.scenarios = scenarios
        # Add constraint on best current lower bound
        if self.vLB >= self.previous_vLB:
            self.deter_model.grb_model.addConstr(
                self.deter_model.grb_objective >= self.vLB)
        self.previous_vLB = copy(self.vLB)
        # Solve model and read solution
        self.deter_model.solve()
        if self.deter_model.grb_model.getAttr(GRB.Attr.Status) == 3:
            return None, -1e6
        else:
            x = self.deter_model.get_var_x_val()
            v = self.deter_model.get_obj_val()
            return x, v

    #   - - - Public methods - - -
    def save_feasible_scenarios(self, feasible_scenarios):
        """
        Saves the scenarios that were feasible in the
        last solution of the upper bound problem.
        Also increment the counter of scenario feasibility.
        """
        self.feasible_scenarios_UB = feasible_scenarios
        self.feasibility_counter += feasible_scenarios

    def incumbent_bound(self, candidate_sols, vUB, vLB, verbose=True):
        """
        Search whether a solution in the given list of
        candidate can improve the current lower bound vLB.
        """
        # Skip empty lists
        if not candidate_sols:
            return False, None, None
        else:
            if verbose:
                print('\n * vLB heuristic *')
                print('Searching if there is a feasible '
                      'candidate solution that improves vLB.')
                print('Nb candidate solutions: ', len(candidate_sols))

        found_best_incumbent, x, obj = self._find_best_incumbent(
                    candidate_sols, vUB, vLB)
        if found_best_incumbent:
            if verbose:
                print('Found an improving solution: increasing vLB from ',
                      vLB, ' to ', obj)
            return found_best_incumbent, x, obj
        else:
            if verbose:
                print('Could not improve vLB using candidate solutions.')
            return False, None, None

    def deterministic_bound(self, xUB):
        """
        Solve lower-bound problem and get solution and cost.
        """
        self.xUB = xUB
        index_scenarios = self._get_index_scenarios()
        selected_scenarios = self._get_selected_scenarios(index_scenarios)
        # Solve lower bound model: a deterministic model with all the
        # selected_scenarios
        x, v = self._solve_deter_model(selected_scenarios)
        return x, v
