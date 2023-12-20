import unittest
import numpy as np

from src.solver.Solver import Solver
from src.instance.ChanceKnapInstance import ChanceKnapInstance


class test_Solver(unittest.TestCase):
    file_location = "./tests/files-for-tests/ccmknap-6-10-10.csv"
    epsilon = 0.2
    chance_instance = ChanceKnapInstance(file_location, True, epsilon)

    def test_initialize(self):
        Solver(self.chance_instance, 1800, 0.1)

    def test_compute_gap(self):
        solver = Solver(self.chance_instance, 1800, 0.1)
        solver.vUB = 102
        solver.vLB = 100
        self.assertEqual(solver._compute_gap(), 0.02)
        solver.vUB = np.inf
        solver.vLB = 100
        self.assertEqual(solver._compute_gap(), np.inf)

    def test_available_time(self):
        solver = Solver(self.chance_instance, 1800, 0.1)
        time = solver._available_time()
        self.assertTrue(isinstance(time, float))

    def test_solve_cclp_model(self):
        solver = Solver(self.chance_instance, 1800, 0.1)
        solver.solve_cclp_model(self.chance_instance, None, use_big_M=False)
