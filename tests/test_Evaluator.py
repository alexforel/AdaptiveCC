import unittest

from src.Evaluator import Evaluator
from src.instance.ChanceKnapInstance import ChanceKnapInstance


class test_Evaluator(unittest.TestCase):
    file_location = "./tests/files-for-tests/ccmknap-6-10-30.csv"
    epsilon = 0.2
    chance_instance = ChanceKnapInstance(file_location, True, epsilon)

    def test_initialize(self):
        Evaluator(self.chance_instance)

    def test_subset_cost(self):
        evaluator = Evaluator(self.chance_instance)
        evaluator.subset_cost([0])

    def test_subset_cost_decreases(self):
        evaluator = Evaluator(self.chance_instance)
        nb_scenarios = self.chance_instance.get_nb_scenarios()
        subset = [0]
        subset_cost, _ = evaluator.subset_cost(subset)
        for i in range(1, nb_scenarios):
            subset.append(i)
            new_cost, _ = evaluator.subset_cost(subset)
            self.assertGreaterEqual(subset_cost, new_cost)
            subset_cost = new_cost

    def test_scenario_costs(self):
        evaluator = Evaluator(self.chance_instance)
        costs = evaluator.scenario_costs([0, 1])
        self.assertEqual(len(costs), 2)
        self.assertEqual(costs[0], evaluator.subset_cost([0])[0])

    def test_partition_cost(self):
        evaluator = Evaluator(self.chance_instance)
        costs, sols = evaluator.partition_cost([[0, 1], [2, 3]])
        self.assertEqual(len(costs), 2)
        self.assertEqual(len(sols), 2)
