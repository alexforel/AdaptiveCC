from gurobipy import GRB
import gurobipy as gp

from src.optim.OptiModel import OptiModel


class AccObjModel(OptiModel):
    """
    Single-level optimization problem
    to minimize the maximum subset cost post-refinement.
    """

    def __init__(self, chance_instance, scenarios, infeasible_scenarios):
        super().__init__(chance_instance, "AccObjModel")
        self.scenarios = scenarios
        self.nb_scenarios = len(scenarios)
        self.infeasible_scenarios = infeasible_scenarios

    #   - - - Private methods - - -
    def _initialize_alpha_obj(self):
        """Initialize objective: minimize unbounded continuous variable."""
        self.var_alpha = self.grb_model.addVar(
            vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
            ub=GRB.INFINITY, name="alpha")
        self.obj = self.grb_model.setObjective(self.var_alpha, GRB.MINIMIZE)

    def _initialize_var_pi(self):
        """Adds the binary assignment variables "pi" to the Gurobi model."""
        self.left_var_pi = self.grb_model.addVars(
            self.nb_scenarios, vtype=GRB.BINARY, name="left_pi")
        self.right_var_pi = self.grb_model.addVars(
            self.nb_scenarios, vtype=GRB.BINARY, name="right_pi")

    def _initialize_var_lambda(self):
        """
        Adds the scenario-independent dual
        variables "lambda" to the Gurobi model.
        """
        nb_vars = self.chance_instance.get_nb_vars()
        self.left_var_lambda = self.grb_model.addVars(
            2 * nb_vars, lb=0.0, ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS, name="left_lambda")
        self.right_var_lambda = self.grb_model.addVars(
            2 * nb_vars, lb=0.0, ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS, name="right_lambda")

    def _initialize_var_eta(self):
        """
        Adds the scenario-dependent dual variables "eta" to the Gurobi model.
        """
        self.left_var_eta = dict()
        self.right_var_eta = dict()
        # Add dual variables vector for each scenario
        for s in range(self.nb_scenarios):
            nb_constraints = int(self.chance_instance.get_nb_constraints(
                self.scenarios[s]))
            self.left_var_eta[s] = self.grb_model.addVars(
                nb_constraints, lb=0.0, ub=GRB.INFINITY,
                vtype=GRB.CONTINUOUS, name="left_eta_s"+str(s))
            self.right_var_eta[s] = self.grb_model.addVars(
                nb_constraints, lb=0.0, ub=GRB.INFINITY,
                vtype=GRB.CONTINUOUS, name="right_eta_s"+str(s))

    def _add_dual_cost_b_vectors_constraints(
            self, scenarios, var_lambda, var_eta):
        nb_vars = self.chance_instance.get_nb_vars()
        #    b_X
        b_X = [0] * nb_vars + [1] * nb_vars
        rhs = gp.quicksum(b_X[k] * var_lambda[k]
                          for k in range(len(b_X)))
        #   Add scenario product
        for s in range(self.nb_scenarios):
            b = self.chance_instance.get_vector_b(scenarios[s])
            nb_constraints = self.chance_instance.get_nb_constraints(
                self.scenarios[s])
            assert len(b) == len(var_eta[s])
            rhs += gp.quicksum(b[i] * var_eta[s][i]
                               for i in range(nb_constraints))
        # Add constraint
        self.grb_model.addConstr(self.var_alpha >= rhs)

    def _add_primal_cost_A_matrices_constraints(
            self, scenarios, var_lambda, var_eta):
        nb_vars = self.chance_instance.get_nb_vars()
        vector_c = self.chance_instance.get_vector_c()
        for j in range(nb_vars):
            lhs = - var_lambda[j] + var_lambda[j+nb_vars]
            #   Add scenario product
            for s in range(self.nb_scenarios):
                A = self.chance_instance.get_matrix_A(scenarios[s])
                nb_constraints = self.chance_instance.get_nb_constraints(
                    self.scenarios[s])
                lhs += gp.quicksum(A[i, j] * var_eta[s][i]
                                   for i in range(nb_constraints))
            # Add constraint
            self.grb_model.addConstr(lhs == vector_c[j])

    def _add_dual_feasibility_constraints(self, scenarios):
        # - Constraint on dual cost and b vectors -
        self._add_dual_cost_b_vectors_constraints(
            scenarios, self.left_var_lambda, self.left_var_eta)
        self._add_dual_cost_b_vectors_constraints(
            scenarios, self.right_var_lambda, self.right_var_eta)
        # - Constraint on primal cost and A matrices -
        self._add_primal_cost_A_matrices_constraints(
            scenarios, self.left_var_lambda, self.left_var_eta)
        self._add_primal_cost_A_matrices_constraints(
            scenarios, self.right_var_lambda, self.right_var_eta)

    def _add_scenario_feasibility_constraints(self, scenarios):
        """
        Add indicator constraints: If a scenario is assigned, it is feasible.
        """
        for s in range(self.nb_scenarios):
            nb_constraints = self.chance_instance.get_nb_constraints(
                scenarios[s])
            for i in range(nb_constraints):
                self.grb_model.addConstr(
                    (self.left_var_pi[s] == 0)
                    >> (self.left_var_eta[s][i] == 0))
                self.grb_model.addConstr(
                    (self.right_var_pi[s] == 0)
                    >> (self.right_var_eta[s][i] == 0))

    def _add_assignment_constraints(self, scenarios, infeasible_scenarios):
        """Add constraints to assign scenarios to the two subsets."""
        self._add_scenario_feasibility_constraints(scenarios)
        # Constraint: A scenario has to be assigned to at least one subset
        for s in range(self.nb_scenarios):
            self.grb_model.addConstr(
                self.left_var_pi[s] + self.right_var_pi[s] == 1)
        # Constraint: There is at least one infeasible scenario per subset
        self.grb_model.addConstr(
                gp.quicksum(self.left_var_pi[s]
                            for s in range(self.nb_scenarios)
                            if scenarios[s] in infeasible_scenarios) >= 1)
        self.grb_model.addConstr(
                gp.quicksum(self.right_var_pi[s]
                            for s in range(self.nb_scenarios)
                            if scenarios[s] in infeasible_scenarios) >= 1)

    #   - - - Public methods - - -
    def build(self, verbose=False):
        """Build the Gurobi model: add all variables and constraints."""
        if not verbose:
            self.grb_model.Params.LogToConsole = 0
        # Initialize objective
        self._initialize_alpha_obj()
        # Initialize binaru assignment and dual decision variables
        self._initialize_var_pi()
        self._initialize_var_lambda()
        self._initialize_var_eta()
        # Add all constraints
        self._add_dual_feasibility_constraints(self.scenarios)
        self._add_assignment_constraints(self.scenarios,
                                         self.infeasible_scenarios)

    def get_subsets(self):
        """Read assignment of solved model."""
        left_subset = []
        right_subset = []
        for s in range(self.nb_scenarios):
            left_pi = self.left_var_pi[s].getAttr(GRB.Attr.X)
            if 0.9 <= left_pi <= 1.1:
                left_subset.append(self.scenarios[s])
            else:
                right_subset.append(self.scenarios[s])
        return left_subset, right_subset
