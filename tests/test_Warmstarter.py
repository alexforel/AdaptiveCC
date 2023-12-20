import unittest
from operator import getitem

from src.Warmstarter import Warmstarter


class test_Warmstarter(unittest.TestCase):

    def test_Warmstarter_initialize(self):
        Warmstarter()

    def test_get_z_start_no_deleted_subsets(self):
        warmstarter = Warmstarter()
        zUB = [0, 1, 1, 0, 1, 0, 1, 2]
        # Test with no deleted subset
        z_start = warmstarter.get_z_start(zUB, [])
        for i in range(len(zUB)):
            if zUB[i] == 1:
                self.assertEqual(z_start[i], 1)
            else:
                self.assertRaises(KeyError, getitem, z_start, i)

    def test_get_z_start_with_deleted_subsets(self):
        warmstarter = Warmstarter()
        # Test with no deleted subset
        zUB = [0, 1, 1, 0]
        deleted_subsets = [2]
        z_start = warmstarter.get_z_start(zUB, deleted_subsets)
        self.assertRaises(KeyError, getitem, z_start, 0)
        self.assertEqual(z_start[1], 1)
        self.assertRaises(KeyError, getitem, z_start, 2)
