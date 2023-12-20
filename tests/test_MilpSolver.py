import unittest
from copy import copy
from parameterized import parameterized

from src.solver.MilpSolver import MilpSolver
from src.instance.ChanceKnapInstance import ChanceKnapInstance


class test_MilpSolver(unittest.TestCase):
    file_location = "./tests/files-for-tests/ccmknap-6-10-100-1.csv"
    epsilon = 0.2
    chance_instance = ChanceKnapInstance(file_location, True, epsilon)

    def test_initialize(self):
        MilpSolver(self.chance_instance)

    fileNames = ["./tests/files-for-tests/ccmknap-6-10-5.csv",
                 "./tests/files-for-tests/ccmknap-6-10-10.csv",
                 "./tests/files-for-tests/ccmknap-6-10-30.csv",
                 "./tests/files-for-tests/ccmknap-6-10-50-1.csv",
                 "./tests/files-for-tests/ccmknap-15-10-10-1.csv",
                 "./tests/files-for-tests/ccmknap-6-10-100-1.csv",
                 "./tests/files-for-tests/ccmknap-10-10-100-1.csv"]

    @parameterized.expand(fileNames)
    def test_compute_big_m(self, filename):
        chance_instance = ChanceKnapInstance(filename, True, self.epsilon)
        solver = MilpSolver(chance_instance)
        # Check only correct inputs are accepted
        with self.assertRaises(ValueError):
            solver._compute_big_m(big_m_method="asdf")
        # Compare big M from Ahmed-Belotti with naive big M
        solver._compute_big_m(big_m_method="naive")
        naiveBigM = copy(solver.big_m_finder.bigM)
        solver._compute_big_m(big_m_method="belotti")
        belottiBigM = copy(solver.big_m_finder.bigM)
        solver._compute_big_m(big_m_method="song")
        songBigM = copy(solver.big_m_finder.bigM)
        # Compare the two big M's elementwise
        for i in range(len(naiveBigM)):
            for s in range(len(naiveBigM[i])):
                self.assertGreaterEqual(naiveBigM[i][s],
                                        belottiBigM[i][s])
                self.assertGreaterEqual(naiveBigM[i][s],
                                        songBigM[i][s])

    @parameterized.expand(fileNames)
    def test_solve_cclp_model(self, filename):
        chance_instance = ChanceKnapInstance(filename, True, self.epsilon)
        solver = MilpSolver(chance_instance)
        solver.solve()
