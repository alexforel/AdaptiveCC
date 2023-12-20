import gurobipy as gp
from gurobipy import GRB

from src.optim.OptiModel import OptiModel


class BigMKnapsackModel(OptiModel):
    """
    Optimization model to find big-M parameter:
    computes the maximum violation of a given (scenario, constraint)
    pair that has a cost within the given lower and upper bounds.

    The model is taken from:
        Belotti, P., Bonami, P., Fischetti, M., Lodi, A., Monaci, M.,
        Nogales-Gómez, A., & Salvagnin, D. (2016). On handling indicator
        constraints in mixed integer programming. Computational Optimization
        and Applications, 65, 545-566.
    and referred to as "Belotti Big-M" in the paper.
    In our simulation, we only use this model when the variables x
    are binary, since the continuous version of the problem can be solved
    using sorting operations.
    """

    def __init__(self, chance_instance, vUB):
        super().__init__(chance_instance, "BigMKnapsackModel")
        self.vUB = vUB

    def _initialize_knapsack_ub(self):
        """Creates the upper bound knapsack constraint."""
        nb_vars = self.chance_instance.get_nb_vars()
        c = self.chance_instance.get_vector_c()
        lhs = gp.quicksum(c[j]*self.var_x[j] for j in range(nb_vars))
        self.knapsack_ub_constraint = self.grb_model.addConstr(
            lhs <= self.vUB)

    #   - - - Public methods - - -
    def initialize_violation_objective(self, s, i):
        """
        Create the objective function:
        maximize violation of constraint (s,i).
        """
        nb_vars = self.chance_instance.get_nb_vars()
        A = self.chance_instance.get_matrix_A(s)
        b = self.chance_instance.get_vector_b(s)
        grb_objective = gp.quicksum(A[i, j]*self.var_x[j]
                                    for j in range(nb_vars)) - b[i]
        self.obj = self.grb_model.setObjective(grb_objective, GRB.MAXIMIZE)

    def build(self, verbose=False):
        """Initialize the Gurobi model."""
        if not verbose:
            self.grb_model.Params.LogToConsole = 0
        self._initialize_var_x()
        self._initialize_knapsack_ub()
