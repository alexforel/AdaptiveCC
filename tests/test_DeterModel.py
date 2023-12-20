import unittest

from src.optim.DeterModel import DeterModel
from src.instance.ChanceKnapInstance import ChanceKnapInstance


class test_DeterModel(unittest.TestCase):
    # Read chance instance from data
    file_location = "./tests/files-for-tests/ccmknap-6-10-5.csv"
    epsilon = 0.2
    chance_instance = ChanceKnapInstance(file_location, True, epsilon)

    def test_Initialize(self):
        DeterModel(self.chance_instance)

    def test_Build(self):
        detModel = DeterModel(self.chance_instance)
        detModel.build(0)
        detModel.build(1)
        detModel.build(2)
        detModel.build(3)
        detModel.build(4)
        # With error
        with self.assertRaises(IndexError):
            detModel.build(5)
            detModel.build(6)

    def test_Solve(self):
        self.nb_scenarios = self.chance_instance.get_nb_scenarios()
        for s in range(self.nb_scenarios):
            detModel = DeterModel(self.chance_instance)
            detModel.build(s)
            detModel.solve()

    def test_ModelOutput(self):
        nb_scenarios = self.chance_instance.get_nb_scenarios()
        # Unbounded model
        detModel = DeterModel(self.chance_instance)
        detModel.build([])
        detModel.solve()
        objUnbounded = detModel.get_obj_val()
        # Model with all constraints of all scenario
        detModel = DeterModel(self.chance_instance)
        detModel.build(list(range(nb_scenarios)))
        detModel.solve()
        allScenariosObj = detModel.get_obj_val()
        # Model with a single scenario constraint
        for s in range(nb_scenarios):
            detModel = DeterModel(self.chance_instance)
            detModel.build(s)
            detModel.solve()
            # Test unbounded model always has higher objective
            self.assertTrue(detModel.get_obj_val() < objUnbounded)
            # Test objective with all constraint is smaller
            self.assertTrue(detModel.get_obj_val() > allScenariosObj)
