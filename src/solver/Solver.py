import numpy as np
import time
from gurobipy import GRB

from src.TimeManager import TimeManager
from src.optim.CCLPModel import CCLPModel
from src.Evaluator import Evaluator


class Solver():
    """
    Meta class for all solver-based solution methods to the
    chance-constrained problem.
    """

    def __init__(self, chance_instance, time_limit, gap):
        self.vUB = np.inf
        self.vLB = -np.inf
        self.time_limit = time_limit
        self.gap = gap
        TimeManager.set_limit_and_start_time(time_limit)
        self.chance_instance = chance_instance
        self.evaluator = Evaluator(self.chance_instance)

    #   - - - Private methods - - -
    def _compute_gap(self):
        """Computes the gap using vLB and vUB.

        Return:
            gap (float): the optimality gap
        """
        if self.vLB == -np.inf or self.vUB == np.inf:
            gap = np.inf
        else:
            gap = (self.vUB - self.vLB)/self.vLB
        return gap

    def _available_time(self):
        """Computes the time left before termination."""
        return TimeManager.get_remaining_time()

    def _get_computation_details(self, decimal_places=3):
        """Collects instance and computational details."""
        # Preparing decimal places string
        str_decimal_place = "{:."+str(decimal_places)+"f}"
        # Extracting instance data
        original_file = self.chance_instance.get_file_name()
        nb_vars = self.chance_instance.get_nb_vars()
        var_type = self.chance_instance.get_var_type()[0]
        nb_scenarios = self.chance_instance.get_nb_scenarios()
        nb_constraints = self.chance_instance.get_nb_constraints(0)
        epsilon = str_decimal_place.format(self.chance_instance.get_epsilon())
        # Extracting computation data
        final_computation_time = str_decimal_place.format(
            TimeManager.get_total_time())
        final_gap = str_decimal_place.format(self._compute_gap()*100.0)
        # Preparing list to be printed
        instance_details = [original_file, nb_vars, nb_constraints,
                            nb_scenarios, var_type*1, epsilon]
        computation_details = [final_computation_time,
                               str_decimal_place.format(self.vLB),
                               str_decimal_place.format(self.vUB),
                               final_gap]
        return instance_details, computation_details

    #   - - - Public methods - - -
    def write_all_computation_details(self, output_file_location):
        """Writes all computational info to output_file_location."""
        raise NotImplementedError

    def solve_cclp_model(self, chance_instance_part,
                         bigMFinder,
                         z_start=None,
                         prune_indices=[],
                         time_limit=1800,
                         elapsed_time=None,
                         gap=1e-4,
                         use_one_thread=True,
                         use_big_M=False,
                         use_lazy=False,
                         save_bounds=False,
                         path=None,
                         verbose=False):
        """
        Instantiate and solve a chance-constrained problem.
        This function is used equivalently for partitioned and original
        problems depending on whether the first input is a chance_instance
        or a chance_instance_part.
        """
        start_build = time.time()
        cclp_model = CCLPModel(chance_instance_part, bigMFinder)
        cclp_model.build(use_big_M=use_big_M, use_lazy=use_lazy,
                         verbose=verbose, z_start=z_start)
        # - Pruning -
        # Set the indicator variables to 0 for the given indices
        if len(prune_indices) > 0:
            print('Pruning ', len(prune_indices),
                  ' subsets thanks to lower bound.')
            for i in prune_indices:
                cclp_model.fix_z_to_zero(i)

        # - Set parameters of Gurobi -
        if use_one_thread:
            cclp_model.grb_model.setParam(GRB.Param.Threads, 1)
        cclp_model.grb_model.setParam(GRB.Param.IntFeasTol, 1e-9)
        cclp_model.grb_model.setParam(GRB.Param.FeasibilityTol, 1e-9)
        cclp_model.grb_model.setParam(GRB.Param.MIPGap, gap)

        # Manage time left for solver
        end_build = time.time()
        real_time_limit = time_limit - (end_build - start_build)
        print('Time left: ', real_time_limit)
        cclp_model.grb_model.setParam(GRB.Param.TimeLimit, real_time_limit)

        # - Solve and read results -
        cclp_model.solve(save_bounds=save_bounds, path=path,
                         elapsed_time=elapsed_time)
        x = cclp_model.get_var_x_val()
        z = cclp_model.get_var_z_val()
        v_obj = cclp_model.get_obj_val()
        v_bnd = cclp_model.get_obj_bnd()

        return x, z, v_obj, v_bnd
