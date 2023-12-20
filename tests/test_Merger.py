import unittest

from src.Merger import Merger
from src.instance.ChanceKnapInstance import ChanceKnapInstance
from src.solver.AdaptivePartitioner import AdaptivePartitioner


class test_Merger(unittest.TestCase):
    # Read chance instance from data
    file_location = "./tests/files-for-tests/ccmknap-6-10-10.csv"
    epsilon = 0.2
    chance_instance = ChanceKnapInstance(file_location, True, epsilon)
    method = AdaptivePartitioner(chance_instance)

    def test_init(self):
        Merger(self.chance_instance, self.method.chance_instance_part)

    def test_single_subset_margin(self):
        merger = Merger(self.chance_instance, self.method.chance_instance_part)
        subset_margin = merger._single_subset_margin(0, [0, 0, 0, 0, 0, 0])
        self.assertGreater(subset_margin, 0.)

    def test_merge_two_subsets_in_partition(self):
        merger = Merger(self.chance_instance, self.method.chance_instance_part)
        merger.target_subsets = []
        # Test simple merge
        merger.deleted_subsets = []
        partition = merger._merge_two_subsets_in_partition(
            [[0, 1], [2, 3]], 0, 1)
        partition = merger._delete_subsets(partition)
        self.assertEqual(partition, [[0, 1, 2, 3]])
        merger.deleted_subsets = []
        partition = merger._merge_two_subsets_in_partition(
            [[0, 1], [2, 3], [4]], 0, 1)
        partition = merger._delete_subsets(partition)
        self.assertEqual(partition, [[0, 1, 2, 3], [4]])
        merger.deleted_subsets = []
        partition = merger._merge_two_subsets_in_partition(
            [[0, 1], [2, 3], [4]], 0, 2)
        partition = merger._delete_subsets(partition)
        self.assertEqual(partition, [[0, 1, 4], [2, 3]])

    def test_merger(self):
        merger = Merger(self.chance_instance, self.method.chance_instance_part)
        partition = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        xUB = [0, 0, 0, 0, 0, 0]
        candidate_subsets = [0, 1, 2]
        # Test can merge two subsets
        mu = 1
        _, _ = merger.merge_feasible(partition, xUB, candidate_subsets, mu)
        for c in merger.target_subsets:
            self.assertTrue(c in [0, 1, 2])
            self.assertTrue(c not in [3, 4])
        self.assertTrue(set(merger.deleted_subsets).issubset([0, 1, 2]))
        for i in merger.deleted_subsets:
            self.assertTrue(i not in [3, 4])
        # Test can merge three subsets
        mu = 2
        _, _ = merger.merge_feasible(partition, xUB, candidate_subsets, mu)
        for c in merger.target_subsets:
            self.assertTrue(c in [0, 1, 2])
            self.assertTrue(c not in [3, 4])
        self.assertEqual(set(merger.deleted_subsets), set([1, 2]))
        for i in merger.deleted_subsets:
            self.assertTrue(i not in [3, 4])
