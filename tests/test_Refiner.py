import unittest

from src.refiner.Refiner import Refiner
from src.instance.ChanceKnapInstance import ChanceKnapInstance


class test_Refiner(unittest.TestCase):
    # Read chance instance from data
    file_location = "./tests/files-for-tests/ccmknap-6-10-10.csv"
    epsilon = 0.2

    def test_Initialize(self):
        chance_instance = ChanceKnapInstance(
            self.file_location, True, self.epsilon)
        Refiner(chance_instance, False)

    def test_count_infeasible_scenarios(self):
        chance_instance = ChanceKnapInstance(
            self.file_location, True, self.epsilon)
        refiner = Refiner(chance_instance, False)
        refiner.partition = [[1, 4, 5], [2, 3]]
        # Test a few instances
        infScenarios = [5, 2, 3]
        nb_inf_scenarios = refiner._count_infeasible_scenarios(
            refiner.partition, infScenarios)
        self.assertTrue(nb_inf_scenarios == [1, 2])
        infScenarios = []
        nb_inf_scenarios = refiner._count_infeasible_scenarios(
            refiner.partition, infScenarios)
        self.assertTrue(nb_inf_scenarios == [0, 0])
        infScenarios = [7, 8]
        nb_inf_scenarios = refiner._count_infeasible_scenarios(
            refiner.partition, infScenarios)
        self.assertTrue(nb_inf_scenarios == [0, 0])

    def test_find_infeasible_subsets(self):
        chance_instance = ChanceKnapInstance(
            self.file_location, True, self.epsilon)
        refiner = Refiner(chance_instance, False)
        refiner.partition = [[1, 4, 5], [2, 3], [6, 7]]
        # Test a few instances
        # - Instance 1
        refiner.infeasible_scenarios = [5, 2, 3]
        feasible_subsets, infeasible_subsets = (
            refiner._find_infeasible_subsets(refiner.partition))
        self.assertTrue(2 in feasible_subsets)
        self.assertTrue(0 in infeasible_subsets)
        self.assertTrue(1 in infeasible_subsets)
        self.assertTrue(refiner.nb_inf_scenarios == [1, 2, 0])
        # - Instance 2
        refiner.infeasible_scenarios = []
        feasible_subsets, infeasible_subsets = (
            refiner._find_infeasible_subsets(refiner.partition))
        self.assertTrue(2 in feasible_subsets)
        self.assertTrue(0 in feasible_subsets)
        self.assertTrue(1 in feasible_subsets)
        self.assertTrue(refiner.nb_inf_scenarios == [0, 0, 0])

    def test_find_number_of_splits(self):
        chance_instance = ChanceKnapInstance(
            self.file_location, True, self.epsilon)
        refiner = Refiner(chance_instance, False)
        # Note: epsilon = 0.2, nb_scenarios = 10
        refiner.infeasible_subsets = [0, 1]
        mu = refiner._find_number_of_splits_to_perform(2)
        self.assertEqual(mu, 1)
        refiner.infeasible_subsets = [0]
        mu = refiner._find_number_of_splits_to_perform(1)
        self.assertEqual(mu, 2)
        refiner.infeasible_subsets = []
        mu = refiner._find_number_of_splits_to_perform(0)
        self.assertEqual(mu, 3)
        with self.assertRaises(AssertionError):
            refiner.infeasible_subsets = [0, 1, 3]
            mu = refiner._find_number_of_splits_to_perform(3)

    def test_split_sorted_subset(self):
        chance_instance = ChanceKnapInstance(
            self.file_location, True, self.epsilon)
        refiner = Refiner(chance_instance, False)
        # Test a few instances
        scenarios = [0, 1, 2, 3, 4, 5]
        even_subset, odd_subset = refiner._split_sorted_subset(scenarios)
        self.assertEqual(even_subset, [0, 2, 4])
        self.assertTrue(odd_subset, [1, 3, 5])
        scenarios = [5, 3, 2, 1]
        even_subset, odd_subset = refiner._split_sorted_subset(scenarios)
        self.assertEqual(even_subset, [5, 2])
        self.assertTrue(odd_subset, [3, 1])
        scenarios = [5, 3]
        even_subset, odd_subset = refiner._split_sorted_subset(scenarios)
        self.assertEqual(even_subset, [5])
        self.assertTrue(odd_subset, [3])
        # Test cannot split on a single or empty subset
        with self.assertRaises(AssertionError):
            refiner._split_sorted_subset([1])
            refiner._split_sorted_subset([])

    def test_split_scenario_subset(self):
        chance_instance = ChanceKnapInstance(
            self.file_location, True, self.epsilon)
        refiner = Refiner(chance_instance, False)
        refiner.partition = [[1, 4, 5, 8, 9], [0, 2, 3], [6, 7]]
        refiner.is_scenario_infeasible = [
            0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
        refiner._split_scenarios_subset([1, 4, 5, 8, 9], 0)
        self.assertEqual(refiner.partition[0], [1, 5, 9])
        self.assertEqual(refiner.partition[1], [0, 2, 3])
        self.assertEqual(refiner.partition[2], [6, 7])
        self.assertEqual(refiner.partition[3], [4, 8])
