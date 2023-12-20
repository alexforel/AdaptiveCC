import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
import csv


class OptiModel():
    """Metaclass for optimization models."""
    # Use single Gurobi environment for all optimization models:
    # avoid querying a license each time a model is solved
    env = gp.Env()

    def __init__(self, chance_instance, modelName):
        self.chance_instance = chance_instance
        self.nb_scenarios = self.chance_instance.get_nb_scenarios()
        # Create gurobi model with name and Gurobi environment
        self.grb_model = gp.Model(modelName, env=self.env)

    #   - - - Private methods - - -
    def _initialize_var_x(self):
        """Adds the decision variables "x" to the Gurobi model."""
        # Get data from chance instance class
        nb_vars = self.chance_instance.get_nb_vars()
        var_type = self.chance_instance.get_var_type()
        var_lb = self.chance_instance.get_var_lb()
        var_ub = self.chance_instance.get_var_ub()
        # Define the gurobi var type based on the getter
        self.grb_var_type = np.array([GRB.CONTINUOUS if var_type[i]
                                      else GRB.BINARY for i in range(nb_vars)])
        # Create x variables
        self.var_x = self.grb_model.addVars(nb_vars, lb=var_lb, ub=var_ub,
                                            vtype=self.grb_var_type, name="x")

    def _read_scenario_data(self, scenario):
        """
        Read all information on the uncertain constraints for a
        given scenario.
        'A' is a matrix and 'b' a vector and all constraints
        have the form: A x <= b.
        """
        A = self.chance_instance.get_matrix_A(scenario)
        b = self.chance_instance.get_vector_b(scenario)
        nb_constraints = self.chance_instance.get_nb_constraints(scenario)
        nb_vars = self.chance_instance.get_nb_vars()
        return A, b, nb_constraints, nb_vars

    def _add_single_bigM_constraint(self, lhs_i, b_i, bigM_i, var_z):
        """
        Add bigM indicator constraint for the i-th constraint of
        the s-th scenario.
        """
        return self.grb_model.addConstr(lhs_i <= b_i + bigM_i * (1 - var_z))

    def _add_single_fixed_bigM_constraint(self, lhs_i, b_i, bigM_i):
        """
        Add constraint without z variable: the constraint is always satisfied,
        however, keep the constraint in model since it may not be binding when
        using binary x variables.
        """
        return self.grb_model.addConstr(lhs_i <= b_i + bigM_i)

    def _add_bigM_constraint(self, scenario, bigM, use_lazy=False):
        """Add all big M constraints for the input scenario."""
        A, b, nb_constraints, nb_vars = self._read_scenario_data(scenario)
        # Loop over each constraint of the scenario and add them to model
        bigM_cstrts = dict()
        for i in range(nb_constraints):
            lhs_i = self._lhs_constraint(i, A, self.var_x, nb_vars)
            if bigM[i] >= -1e-6:
                bigM_cstrts[i] = self._add_single_bigM_constraint(
                    lhs_i, b[i], bigM[i], self.var_z[scenario])
                if use_lazy:
                    bigM_cstrts[i].Lazy = 1
            else:
                self.skip_counter += 1
                if (self.grb_var_type == GRB.CONTINUOUS).all():
                    # We do not need to implement this constraint!
                    pass
                else:
                    # Add constraint with fixed value for z and negative bigM
                    bigM_cstrts[i] = self._add_single_fixed_bigM_constraint(
                        lhs_i, b[i], bigM[i])
                    if use_lazy:
                        bigM_cstrts[i].Lazy = 1
        return bigM_cstrts

    def _add_all_bigM_constraints(self, use_lazy=False):
        """Adds all bigM constraints of all scenario to the Gurobi model."""
        # Loop over all scenarios and call the private function
        #    to add the constraints
        self.bigM_constraints = dict()
        self.feasibility_constraints = dict()
        self.skip_counter = 0
        for s in range(self.nb_scenarios):
            bigM = self.bigMFinder.get_vector_big_M(s)
            self.bigM_constraints[s] = self._add_bigM_constraint(
                s, bigM, use_lazy=use_lazy)
        # Print summary of constraint implementation based on negative bigMs
        if (self.grb_var_type == GRB.CONTINUOUS).all():
            print('Could skip ', self.skip_counter, ' constraints'
                  ' without z variable thanks to negative big M.')
        else:
            print('Could setup ', self.skip_counter, ' constraints'
                  ' without z variable thanks to negative big M.')

    def _add_chance_constraint(self):
        """Add the chance constraint to the model."""
        # Get data from chance instance class
        proba = self.chance_instance.get_proba()
        epsilon = self.chance_instance.get_epsilon()
        # The chance constraint is added to the model using gurobi quicksum
        self.chance_constraint = self.grb_model.addConstr(
            gp.quicksum(self.var_z[s]*proba[s]
                        for s in range(self.nb_scenarios)) >= 1-epsilon)

    def _initialize_binary_var_z(self, relax=False):
        """
        Adds the binary indicator variables "z" to the gurobi model
        """
        if relax:
            self.var_z = self.grb_model.addVars(self.nb_scenarios,
                                                vtype=GRB.CONTINUOUS, name="z")
        else:
            self.var_z = self.grb_model.addVars(self.nb_scenarios,
                                                vtype=GRB.BINARY, name="z")

    @staticmethod
    def _lhs_constraint(i, A, x, nb_vars):
        """Create the left-hand side of the constraint using grb quicksum.

        Args:
            i (int): index of constraint
            A (Matrix): matrix of constraints for scenario i
            x (GRB.CONTINUOUS/BINARY): decision variables
            nb_vars (int): number of constraints per scenario

        Returns:
            lhs (GRB.LinExpr): Vector of left-hand side linear expressions
        """
        lhs_i = gp.quicksum(A[i, j]*x[j] for j in range(nb_vars))
        return lhs_i

    def _initialize_obj(self):
        """Adds the objective (c^T * x) to the gurobi model."""
        # Get data from chance instance class
        vector_c = self.chance_instance.get_vector_c()
        nb_vars = self.chance_instance.get_nb_vars()

        # Create objective using grb quicksum
        self.grb_objective = gp.quicksum(
            vector_c[i]*self.var_x[i] for i in range(nb_vars))
        # Set objective
        self.obj = self.grb_model.setObjective(self.grb_objective,
                                               GRB.MAXIMIZE)

    def _save_bounds_callback(self, model, where):
        """
        Save the lower and upper bounds evolution through
        the solving process.

        Adapted from:
        https://support.gurobi.com/hc/en-us/community/
            posts/360067833591-Working-with-the-optimization-log
        """
        if where == gp.GRB.Callback.MIP:
            cur_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
            cur_bd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
            # Did objective value or best bound change?
            if self._obj != cur_obj or self._bd != cur_bd:
                self._obj = cur_obj
                self._bd = cur_bd
                self._data.append(
                    [time.time() - self.start_time + self.elapsed_time,
                     -cur_obj, -cur_bd])

    def _save_bounds_to_file(self, path):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(self._data)

    #   - - - Public methods - - -
    def solve(self, save_bounds=False, path=None, elapsed_time=0.0):
        if save_bounds:
            assert path is not None
            self._obj = None
            self._bd = None
            self._data = []
            self.start_time = time.time()
            self.elapsed_time = elapsed_time

            self.grb_model.optimize(
                lambda model, where:
                    self._save_bounds_callback(model, where))
        else:
            self.grb_model.optimize()

        # If the solve is interrupted, save bounds and
        # propagate interruption.
        if self.grb_model.getAttr(GRB.Attr.Status) == 11:
            if save_bounds:
                self._save_bounds_to_file(path)
            raise KeyboardInterrupt
        if save_bounds:
            self._save_bounds_to_file(path)

    def get_obj_val(self):
        return self.grb_model.ObjVal

    def get_obj_bnd(self):
        return self.grb_model.ObjBound

    def get_var_x_val(self):
        items = range(self.chance_instance.get_nb_vars())
        var_x_val = np.array([self.var_x[j].getAttr(GRB.Attr.X)
                             for j in items])
        return var_x_val

    def write_model_to_file(self, model_file):
        """
        Print gurobi model to a file using the
        built-in write function.
        """
        self.grb_model.write(model_file)
