import unittest
import numpy as np

from src.Initializer import Initializer
from src.instance.ChanceKnapInstance import ChanceKnapInstance
from src.instance.PartitionChanceKnapInstance import \
     PartitionChanceKnapInstance


class test_Initializer(unittest.TestCase):
    # Read chance instance from data
    file_location = "./tests/files-for-tests/ccmknap-6-10-10.csv"
    epsilon = 0.2
    chance_instance = ChanceKnapInstance(file_location, True, epsilon)
    chance_instance_part = PartitionChanceKnapInstance(chance_instance)

    def test_Initialize(self):
        Initializer(self.chance_instance, self.chance_instance_part)

    def test_check_partition(self):
        initializer = Initializer(self.chance_instance,
                                  self.chance_instance_part)
        # Test method passes with correct input
        partition = [list(range(self.chance_instance.get_nb_scenarios()))]
        initializer._check_partitions(partition, 1)
        partition = [[0, 3, 4, 5, 6], [1, 2], [7, 8, 9]]
        initializer._check_partitions(partition, 3)
        with self.assertRaises(AssertionError):
            # Return error if not enough scenarios assigned
            partition = [[0, 3, 4, 5, 6], [1, 2]]
            initializer._check_partitions(partition, 2)
            # Return error if duplicates
            partition = [[0, 3, 4, 5, 6], [1, 2], [7, 7, 7]]
            initializer._check_partitions(partition, 3)
            # Return error if too many scenarios
            partition = [[0, 1, 2], [0, 3, 4, 5, 6], [1, 2], [7, 7, 7]]
            initializer._check_partitions(partition, 4)
            # Return error if scenarios ok but not enough clusters
            partition = [[0, 3, 4, 5, 6], [1, 2], [7, 8, 9]]
            initializer._check_partitions(partition, 4)
            # Return error if scenarios ok but too many clusters
            initializer._check_partitions(partition, 2)

    def test_random_partition(self):
        initializer = Initializer(self.chance_instance,
                                  self.chance_instance_part)
        initializer._random_partition([0, 1, 2, 4, 5], 1)
        initializer._random_partition([0, 1, 2, 4, 5], 2)
        initializer._random_partition([0, 1, 2, 4, 5], 3)
        initializer.create_first_partition(1, partition_type="random")
        initializer.create_first_partition(2, partition_type="random")
        initializer.create_first_partition(3, partition_type="random")
        with self.assertRaises(AssertionError):
            initializer.create_first_partition(300, partition_type="random")

    def _test_get_all_single_scenario_costs(self):
        initializer = Initializer(self.chance_instance,
                                  self.chance_instance_part)
        scenarioList = list(range(self.chance_instance.get_nb_scenarios()))
        costs = initializer._get_all_single_scenario_costs(scenarioList)
        # Test all single-scenario costs are unique
        assert (np.unique(costs).size == len(costs))

    def test_cost_initializer(self):
        initializer = Initializer(self.chance_instance,
                                  self.chance_instance_part)
        scenarioList = list(range(self.chance_instance.get_nb_scenarios()))
        nbClusterList = [1, 2, 3]
        for n in nbClusterList:
            partition1 = initializer._single_scenario_cost_partition(
                scenarioList, n)[0]
            partition2 = initializer.create_first_partition(
                n, partition_type="cost")
            for c in range(n):
                self.assertTrue(partition1[c] == partition2[c])
        with self.assertRaises(AssertionError):
            initializer.create_first_partition(300, partition_type="cost")
