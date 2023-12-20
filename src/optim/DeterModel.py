from src.optim.OptiModel import OptiModel


class DeterModel(OptiModel):
    """Create and solve a deterministic model."""

    def __init__(self, chance_instance):
        super().__init__(chance_instance, "DeterModel")

    #   - - - Private methods - - -
    def _add_constraints(self, scenario):
        """Add the feasibility constraints of a single scenario."""
        A, b, nb_constraints, nb_vars = self._read_scenario_data(scenario)
        # Loop over each constraint of the scenario and add them to model
        feasibility_constraint = dict()
        for i in range(nb_constraints):
            lhs = self._lhs_constraint(i, A, self.var_x, nb_vars)
            feasibility_constraint[i] = self.grb_model.addConstr(
                lhs <= b[i])
        return feasibility_constraint

    def _remove_feasibility_constraints(self, s: int):
        """
        Remove all the feasibility constraints linked to scenario s.

        Args:
            s (int): scenario index
        """
        nb_constraints = self.chance_instance.get_nb_constraints(s)
        for i in range(nb_constraints):
            constraint = self.feasibility_constraint[s][i]
            self.grb_model.remove(constraint)

    #   - - - Public methods - - -
    def build(self, scenarios, verbose=False):
        """Initialize the Gurobi model."""
        if not verbose:
            self.grb_model.Params.LogToConsole = 0
        self._initialize_var_x()
        self._initialize_obj()
        # Build a deterministic smodel
        self.feasibility_constraint = dict()
        self.add(scenarios)

    def add(self, scenarios):
        """
        Add the constraints of given subset(s)/scenario(s).
        The input can be a list or an integer.
        """
        if scenarios.__class__ == list:
            for s in scenarios:
                self.feasibility_constraint[s] = self._add_constraints(s)
        else:
            s = scenarios
            self.feasibility_constraint[s] = self._add_constraints(s)

    def remove(self, scenarios):
        """Remove constraints from previous subset(s)/scenario(s)."""
        if scenarios.__class__ == list:
            for s in scenarios:
                self._remove_feasibility_constraints(s)
        else:
            self._remove_feasibility_constraints(scenarios)
