""" Integration tests:
Verify that all variations of optimization models and partitioners
return the same solution and objective on a given instance.
"""
import unittest
import itertools as itertools
from parameterized import parameterized

from src.BigMFinder import BigMFinder
from src.optim.CCLPModel import CCLPModel
from src.instance.ChanceKnapInstance import ChanceKnapInstance
from src.solver.AdaptivePartitioner import AdaptivePartitioner
from src.solver.MilpSolver import MilpSolver


class test_IntegrationBinary(unittest.TestCase):
    # Read chance instance from data
    file_location = "./tests/files-for-tests/ccmknap-15-10-10-1.csv"
    epsilon = 0.2
    chance_instance = ChanceKnapInstance(file_location, False, epsilon)
    # Baseline model: Gurobi extended formulation with indicator constraints
    cclp_model = CCLPModel(chance_instance, None)
    cclp_model.build(use_big_M=False, verbose=False)
    cclp_model.solve()
    x = cclp_model.get_var_x_val()
    z = cclp_model.get_var_z_val()
    v = cclp_model.get_obj_val()

    def _compare_two_vectors_of_solutions(self, x1, x2):
        """
        Assert that all the elements of two vectors are almost equal.
        """
        for i, x2_i in enumerate(x2):
            self.assertAlmostEqual(x2_i, x1[i])

    def test_cclp_models(self):
        #     - Chance-constrained problem: Gurobi benchmark -
        # Test indicator constraints
        solver = MilpSolver(self.chance_instance)
        solver.solve(use_big_m=False)
        self.assertAlmostEqual(self.v, solver.vLB, places=5)
        self._compare_two_vectors_of_solutions(self.x, solver.xLB)
        # Test naive big M
        solver = MilpSolver(self.chance_instance)
        solver.solve(use_big_m=True, big_m_method="naive")
        self.assertAlmostEqual(self.v, solver.vLB, places=5)
        self._compare_two_vectors_of_solutions(self.x, solver.xLB)
        # Test Belotti big M
        solver = MilpSolver(self.chance_instance)
        solver.solve(use_big_m=True, big_m_method="belotti")
        self.assertAlmostEqual(self.v, solver.vLB, places=5)
        self._compare_two_vectors_of_solutions(self.x, solver.xLB)
        # Test Song big M
        solver = MilpSolver(self.chance_instance)
        solver.solve(use_big_m=True, big_m_method="song")
        self._compare_two_vectors_of_solutions(self.x, solver.xLB)
        self._compare_two_vectors_of_solutions(self.z, solver.zLB)
        self.assertAlmostEqual(self.v, solver.vLB, places=5)

    # Parameters for AdaptivePartitioners test sequence
    initPartitions = ['random', 'cost']
    splitMethods = ['random', 'cost']
    bigMMethods = ["naive", "belotti", "song"]

    @parameterized.expand(
            itertools.product(initPartitions, splitMethods,
                              [True], bigMMethods, [False, True]))
    def test_adaptive_partitioners(self, init_part, split_method,
                                   useBigM, big_m_method, useMerger):
        # Run adaptive partitioner and compare outputs
        if split_method == 'cost':
            USE_ACC_OBJ = True
        else:
            USE_ACC_OBJ = False
        method = AdaptivePartitioner(
            self.chance_instance, split_method=split_method,
            initial_partition_type=init_part,
            projection_method='rescaled_max_violation',
            use_acc_obj=USE_ACC_OBJ)
        partitionBigMFinder = BigMFinder(method.chance_instance_part)
        method.solve(partitionBigMFinder,
                     use_big_M=useBigM, big_m_method=big_m_method,
                     use_merger=useMerger)
        # Compare output of adaptive partitioner and Gurobi baseline
        self.assertAlmostEqual(method.vUB, self.v, places=5)
        self._compare_two_vectors_of_solutions(self.x, method.xUB)
