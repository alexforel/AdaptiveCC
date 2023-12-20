import numpy as np
from gurobipy import GRB

from src.optim.OptiModel import OptiModel


class CCLPModel(OptiModel):
    """Create and solve a CCLP model."""
    def __init__(self, chance_instance, bigMFinder):
        super().__init__(chance_instance, "CCLP")
        self.bigMFinder = bigMFinder

    #   - - - Private methods - - -
    def _warm_start_binary_var_z(self, z_start=None):
        """Warm start z variables.
        If available, use z_start to warm start the binary variables.

        Args:
            z_start (dict, optional): dict of variables to warm start.
                In practice, we only warmstart variables equal to 1.
        """
        # Loop over the scenarios available in z_start and
        # set starting value
        if z_start is not None:
            assert isinstance(z_start, dict)
            for s in z_start:
                self.var_z[s].start = z_start[s]

    def _add_indicator_constraint(self, scenario):
        """Create all indicator constraints of a scenario."""
        A, b, nb_constraints, nb_vars = self._read_scenario_data(scenario)
        # Loop over each constraint of the scenario and add them to model
        ind_constraints = dict()
        for i in range(nb_constraints):
            lhs = self._lhs_constraint(i, A, self.var_x, nb_vars)
            # Add constraint
            ind_constraints[i] = self.grb_model.addGenConstrIndicator(
                self.var_z[scenario], True, lhs <= b[i])
        return ind_constraints

    def _add_all_indicator_constraints(self):
        """Adds all indicator constraints to the gurobi model."""
        # Loop over all scenarios and call
        # the private function to add the constraints
        self.ind_constraints = dict()
        for s in range(self.nb_scenarios):
            self.ind_constraints[s] = self._add_indicator_constraint(s)

    #   - - - Public methods - - -
    def get_var_z_val(self):
        """Returns the values of the z indicator variables."""
        scenarios = range(self.nb_scenarios)
        var_z_val = np.array([self.var_z[s].getAttr(GRB.Attr.X)
                              for s in scenarios])
        return var_z_val

    def fix_z_to_zero(self, s):
        """
        ''Prune'' a scenario or subset by fixing its indicator variable
        to zero: thus, it will never be satisfied.
        """
        self.grb_model.addConstr(self.var_z[s] == 0)

    def build(self, use_big_M=False, use_lazy=False,
              verbose=True, z_start=None):
        """Build CCLP model: add variables, constraints and objective.

        Args:
            use_big_M (bool, optional): if False, use general
                indicator constraints. If True, use Big M. Defaults to False.
            use_lazy (bool, optional): if True, set all big M constraints
                as ''Lazy'' constraints in Gurobi. Defaults to False.
            verbose (bool, optional): if True, print solve process.
                Defaults to True.
            z_start (dict, optional): z variables to warmstart.
                Defaults to None.
        """
        if not verbose:
            self.grb_model.Params.LogToConsole = 0
        self._initialize_var_x()
        self._initialize_obj()
        # Build a chance-constrained model
        #    with indicator variables and constraints
        self._initialize_binary_var_z()
        self._warm_start_binary_var_z(z_start=z_start)
        if use_big_M:
            self._add_all_bigM_constraints(use_lazy=use_lazy)
        else:
            self._add_all_indicator_constraints()
        self._add_chance_constraint()
