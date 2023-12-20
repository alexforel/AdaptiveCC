import unittest

from src.refiner.RandomRefiner import RandomRefiner
from src.instance.ChanceKnapInstance import ChanceKnapInstance


class test_RandomRefiner(unittest.TestCase):
    # Read chance instance from data
    file_location = "./tests/files-for-tests/ccmknap-6-10-10.csv"
    epsilon = 0.2

    def test_Initialize(self):
        chance_instance = ChanceKnapInstance(
            self.file_location, True, self.epsilon)
        RandomRefiner(chance_instance)

    def test_get_sorted_scenarios(self):
        chance_instance = ChanceKnapInstance(
            self.file_location, True, self.epsilon)
        refiner = RandomRefiner(chance_instance)
        # Test a few instances
        inpScen = [0, 1, 2, 3, 4, 5]
        outScen = refiner._get_sorted_scenarios(inpScen)
        self.assertNotEqual(inpScen,  outScen)
        outScen2 = refiner._get_sorted_scenarios(inpScen)
        self.assertNotEqual(inpScen,  outScen)
        self.assertNotEqual(outScen2,  outScen)
        outScen3 = refiner._get_sorted_scenarios(inpScen)
        self.assertNotEqual(inpScen,  outScen)
        self.assertNotEqual(outScen2,  outScen3)
        self.assertNotEqual(outScen3,  outScen)

    def test_refine(self):
        chance_instance = ChanceKnapInstance(
            self.file_location, True, self.epsilon)
        refiner = RandomRefiner(chance_instance)
        partition = [[1, 4, 5, 8, 9], [0, 2, 3], [6, 7]]
        refiner.partition = [[1, 4, 5, 8, 9], [0, 2, 3], [6, 7]]
        refiner.infeasible_scenarios = [[1, 4]]
        refiner.nb_inf_scenarios = [2, 0, 0]
        refiner.is_scenario_infeasible = [
            0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
        # Test splitting
        refiner._split_scenarios_subset([1, 4, 5, 8, 9], 0)
        # Test that the first partition has changed
        self.assertNotEqual(refiner.partition[0],
                            partition[0])
