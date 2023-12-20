import unittest
import numpy as np
import itertools as itertools
from parameterized import parameterized

from src.BigMFinder import BigMFinder
from src.instance.ChanceKnapInstance import ChanceKnapInstance
from violations import compute_all_violations as cpp_song
from src.song_big_m import compute_all_violations as python_song


class test_BigMFinder(unittest.TestCase):
    # Read chance instance from data
    file_location = "./tests/files-for-tests/ccmknap-2-2-5.csv"
    epsilon = 0.2
    chance_instance = ChanceKnapInstance(file_location, True, epsilon)

    @staticmethod
    def _print_differences(k, m, cpp_violations, python_violations):
        for s in range(k):
            for t in range(k):
                for i in range(m):
                    if not np.isclose(cpp_violations[s, t, i],
                                      python_violations[s, t, i]):
                        print('s, t, i:', s, t, i)
                        print('cpp:', cpp_violations[s, t, i])
                        print('python:', python_violations[s, t, i])

    def test_bigMInitialize(self):
        BigMFinder(self.chance_instance)

    def _test_single_naive_big_M(self, bigMFinder, A, b, i, s):
        naiveBigM = (A[i, 0] + A[i, 1] - b[i])
        self.assertEqual(naiveBigM, bigMFinder.bigM[s][i])

    def _assert_all_big_M_equal_naive(self, bigMFinder):
        self.nb_scenarios = self.chance_instance.get_nb_scenarios()
        self.nb_constraints = self.chance_instance.get_nb_constraints(0)
        for s in range(self.nb_scenarios):
            for i in range(self.nb_constraints):
                self.assertEqual(bigMFinder._naive_bigM(s)[i],
                                 bigMFinder.bigM[s][i])

    def test_naiveBigM(self):
        bigMFinder = BigMFinder(self.chance_instance)
        #   - Test manually -
        for s in range(5):
            # Read constraint matrices
            A = self.chance_instance.get_matrix_A(s)
            b = self.chance_instance.get_vector_b(s)
            for i in range(2):
                self._test_single_naive_big_M(bigMFinder, A, b, i, s)
                self._test_single_naive_big_M(bigMFinder, A, b, i, s)
        #   - Test internal function -
        self._assert_all_big_M_equal_naive(bigMFinder)

    def test_belotti_et_al(self):
        file_location = "./tests/files-for-tests/ccmknap-6-10-10.csv"
        epsilon = 0.2
        chance_instance = ChanceKnapInstance(file_location, True, epsilon)
        bigMFinder = BigMFinder(chance_instance)
        upper_bound = 3000
        bigMFinder.run_belotti_et_al_big_M(upper_bound)

    def test_song(self):
        file_location = "./tests/files-for-tests/ccmknap-6-10-10.csv"
        epsilon = 0.2
        chance_instance = ChanceKnapInstance(file_location, True, epsilon)
        bigMFinder = BigMFinder(chance_instance)
        bigMFinder.run_song_et_al_big_m(chance_instance)

    def test_song_python_and_cpp_easy_instances(self):
        k = 10
        m = 10
        n = 10
        # Test on matrices of zeros
        A = np.zeros((k, m, n))
        b = np.zeros((k, m))
        cpp_violations = cpp_song(A, b)
        python_violations = python_song(A, b)
        np.testing.assert_array_almost_equal(cpp_violations, python_violations)
        # Test on matrices of ones
        A = np.ones((k, m, n))
        b = np.ones((k, m))
        A[1, 2, 3] = 10
        b[1, 3] = 5
        cpp_violations = cpp_song(A, b)
        python_violations = python_song(A, b)
        np.testing.assert_array_almost_equal(cpp_violations, python_violations)
        # Test on positive matrix
        A = np.ones((k, m, n))
        b = np.ones((k, m))
        for s in range(k):
            b[s, :] = range(m)
            b[s, :] = b[s, :] * k
        cpp_violations = cpp_song(A, b)
        python_violations = python_song(A, b)
        np.testing.assert_array_almost_equal(cpp_violations, python_violations)

    @parameterized.expand(
            itertools.product(range(10),
                              [(8, 8, 5), (5, 5, 10),  (12, 10, 5)]))
    def test_song_python_and_cpp_random_instances_all_positive(self, i, sizes):
        np.random.seed(i)
        k, m, n = sizes
        A = np.random.rand(k, m, n)
        b = np.random.rand(k, m)
        # Test that two matrices are equal
        cpp_violations = cpp_song(A, b)
        python_violations = python_song(A, b)
        np.testing.assert_array_almost_equal(cpp_violations, python_violations)

    @parameterized.expand(
            itertools.product(range(5), [(8, 8, 5), (5, 5, 10), (10, 10, 8)]))
    def test_song_python_and_cpp_random_instances_with_zeros(self, i, sizes):
        np.random.seed(i)
        k, m, n = sizes
        A = np.random.rand(k, m, n)
        b = np.random.rand(k, m)
        A[A <= 0.5] = 0.0
        b[b <= 0.5] = 0.0
        # Test that two matrices are equal
        cpp_violations = cpp_song(A, b)
        python_violations = python_song(A, b)
        self._print_differences(k, m, cpp_violations, python_violations)
        np.testing.assert_array_almost_equal(cpp_violations,
                                             python_violations)

    fileNames = ["./tests/files-for-tests/ccmknap-6-10-5.csv",
                 "./tests/files-for-tests/ccmknap-6-10-10.csv",
                 "./tests/files-for-tests/ccmknap-6-10-30.csv",
                 "./tests/files-for-tests/ccmknap-15-10-10-1.csv"]

    @parameterized.expand(fileNames)
    def test_song_python_and_cpp_on_instances(self, filename):
        epsilon = 0.2
        chance_instance = ChanceKnapInstance(filename, True, epsilon)
        A = chance_instance.get_matrices_A()
        b = chance_instance.get_vectors_b()
        # Test that two matrices are equal
        cpp_violations = cpp_song(A, b)
        python_violations = python_song(A, b)
        self._print_differences(A.shape[0], A.shape[1],
                                cpp_violations, python_violations)
        np.testing.assert_array_almost_equal(cpp_violations,
                                             python_violations)
