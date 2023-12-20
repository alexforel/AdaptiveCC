"""Integration tests for all lower bound methods."""
import unittest
from parameterized import parameterized

from src.instance.ChanceKnapInstance import ChanceKnapInstance
from src.optim.CCLPModel import CCLPModel
from src.solver.AdaptivePartitioner import AdaptivePartitioner


class test_Integration(unittest.TestCase):
    # Read chance instance from data
    file_location = "./tests/files-for-tests/ccmknap-6-10-10.csv"
    epsilon = 0.2

    def _get_baseline(self, chance_instance):
        # Solve full scenario chance constraint problem
        cclp_model = CCLPModel(chance_instance, None)
        cclp_model.build(use_big_M=False, verbose=False)
        cclp_model.solve()
        x = cclp_model.get_var_x_val()
        z = cclp_model.get_var_z_val()
        v = cclp_model.get_obj_val()
        return x, z, v

    def _compare_two_vectors_of_solutions(self, x, x2):
        for i, x_i in enumerate(x2):
            self.assertAlmostEqual(x_i, x[i])

    # Parameters for AdaptivePartitioners test sequence
    projList = [
        'counter',
        'rescaled_max_violation'
        ]

    @parameterized.expand(projList)
    def test_upper_bound_methods(self, projection_method):
        chance_instance = ChanceKnapInstance(self.file_location, True,
                                             self.epsilon)
        # Chance-constrained problem: Gurobi benchmark -
        x, _, v = self._get_baseline(chance_instance)
        # Run adaptive partitioner and compare outputs
        adaptive_partitioning = AdaptivePartitioner(
            chance_instance, projection_method=projection_method)
        adaptive_partitioning.solve(None)
        self.assertAlmostEqual(adaptive_partitioning.vUB, v)
        self._compare_two_vectors_of_solutions(
            x, adaptive_partitioning.xUB)
