import unittest
import math
import numpy as np

from src.instance.ChanceKnapInstance import ChanceKnapInstance


class test_ChanceKnapInstance(unittest.TestCase):
    """
    Test class for the ChanceKnapInstance class.
    """
    file_location = "./tests/files-for-tests/ccmknap-6-10-5.csv"
    continuous_var = True
    epsilon = 0.3

    def test_initialize(self):
        ChanceKnapInstance(self.file_location,
                           self.continuous_var,
                           self.epsilon)

    def test_indices(self):
        chance_instance = ChanceKnapInstance(self.file_location,
                                             self.continuous_var,
                                             self.epsilon)
        self.assertTrue(chance_instance.get_nb_vars() == 6)
        self.assertTrue(chance_instance.get_nb_scenarios() == 5)

    def test_c_vector(self):
        c_vector = np.array([100.0, 600.0, 1200.0, 2400.0, 500.0, 2000.0])
        chance_instance = ChanceKnapInstance(self.file_location,
                                             self.continuous_var,
                                             self.epsilon)
        object_c_vector = chance_instance.get_vector_c()
        c_equality = np.array_equal(c_vector, object_c_vector)
        self.assertTrue(c_equality)

    def test_b_vectors(self):
        b_vector = np.array([80.0, 96.0, 20.0, 36.0, 44.0,
                             48.0, 10.0, 18.0, 22.0, 24.0])
        chance_instance = ChanceKnapInstance(self.file_location,
                                             self.continuous_var,
                                             self.epsilon)
        nb_scenarios = chance_instance.get_nb_scenarios()
        for s in range(nb_scenarios):
            object_b_vector = chance_instance.get_vector_b(s)
            b_equality = np.array_equal(b_vector, object_b_vector)
            self.assertTrue(b_equality)

    def test_s1_A_matrices(self):
        matrix_A1_list = [
            [8.3724, 10.9111, 14.9003, 78.5605, 19.3014, 0.0],
            [8.0471, 9.5267, 11.9253, 69.1535, 21.1707, 0.0],
            [2.6769, 5.3828, 3.862, 18.9361, 6.0313, 0.0],
            [4.8479, 8.9136, 8.0223, 31.8631, 6.5568, 0.0],
            [5.4217, 14.3308, 6.7009, 35.4751, 6.0885, 0.0],
            [4.7369, 14.1636, 7.5303, 59.9353, 5.0918, 0.0],
            [0.0, 0.0, 0.0, 0.0, 6.8219, 0.0],
            [3.3851, 0.0, 3.9451, 0.0, 8.4066, 0.0],
            [3.3548, 1.6779, 4.0253, 0.0, 8.2171, 0.0],
            [2.6554, 1.9441, 4.5076, 7.7083, 8.8505, 0.0]]
        matrix_A1 = np.round(np.array(matrix_A1_list), decimals=1)
        chance_instance = ChanceKnapInstance(self.file_location,
                                             self.continuous_var,
                                             self.epsilon)
        object_matrix_A1 = chance_instance.get_matrix_A(1)
        A1_equality = np.array_equal(matrix_A1, object_matrix_A1)
        self.assertTrue(A1_equality)

    def test_s3_A_matrices(self):
        matrix_A3_list = [
            [8.1841, 11.7846, 15.5186, 68.3488, 24.6347, 0.0],
            [9.0568, 11.7583, 13.6906, 60.4606, 24.1183, 0.0],
            [3.1744, 6.1058, 3.4227, 17.4788, 6.6504, 0.0],
            [6.0021, 9.5668, 8.043, 30.7466, 4.706, 0.0],
            [5.2378, 14.124, 5.2907, 45.7069, 5.2983, 0.0],
            [5.791, 10.6531, 7.6283, 37.9217, 6.4943, 0.0],
            [0.0, 0.0, 0.0, 0.0, 8.0311, 0.0],
            [3.0788, 0.0, 3.8025, 0.0, 7.7712, 0.0],
            [2.655, 1.9297, 4.0967, 0.0, 7.4852, 0.0],
            [3.4187, 2.0514, 3.7924, 8.468, 6.4113, 0.0]]
        matrix_A3 = np.round(np.array(matrix_A3_list), decimals=1)
        chance_instance = ChanceKnapInstance(self.file_location,
                                             self.continuous_var,
                                             self.epsilon)
        object_matrix_A3 = chance_instance.get_matrix_A(3)
        A3_equality = np.array_equal(matrix_A3, object_matrix_A3)
        self.assertTrue(A3_equality)

    def test_s3_infeasibility(self):
        matrix_A3_list = [
            [8.1841, 11.7846, 15.5186, 68.3488, 24.6347, 0.0],
            [9.0568, 11.7583, 13.6906, 60.4606, 24.1183, 0.0],
            [3.1744, 6.1058, 3.4227, 17.4788, 6.6504, 0.0],
            [6.0021, 9.5668, 8.043, 30.7466, 4.706, 0.0],
            [5.2378, 14.124, 5.2907, 45.7069, 5.2983, 0.0],
            [5.791, 10.6531, 7.6283, 37.9217, 6.4943, 0.0],
            [0.0, 0.0, 0.0, 0.0, 8.0311, 0.0],
            [3.0788, 0.0, 3.8025, 0.0, 7.7712, 0.0],
            [2.655, 1.9297, 4.0967, 0.0, 7.4852, 0.0],
            [3.4187, 2.0514, 3.7924, 8.468, 6.4113, 0.0]]
        matrix_A3 = np.round(np.array(matrix_A3_list), decimals=1)
        b_vector = np.array([80.0, 96.0, 20.0, 36.0, 44.0,
                             48.0, 10.0, 18.0, 22.0, 24.0])
        var_x_val = np.ones(6)
        var_x_val[[0, 3]] = 0
        knap_constraint_qty = len(matrix_A3_list)
        scenario_infeasibility = []
        for i in range(knap_constraint_qty):
            violation = matrix_A3[i, :].dot(var_x_val) - b_vector[i]
            if violation > 1e-7:
                scenario_infeasibility.append(violation)
        chance_instance = ChanceKnapInstance(self.file_location,
                                             self.continuous_var,
                                             self.epsilon)
        object_scenario_inf = chance_instance._get_scenario_infeasibility(
            var_x_val)
        self.assertEqual(object_scenario_inf[3], scenario_infeasibility)

    def test_get_inf_scenarios(self):
        chance_instance = ChanceKnapInstance(self.file_location,
                                             self.continuous_var,
                                             self.epsilon)
        nb_scenarios = 5
        item_qty = 6
        knap_constraint_qty = 10
        var_x_val = np.ones(item_qty)
        var_x_val[[1, 5]] = 0
        is_scenario_infeasible = [False for s in range(nb_scenarios)]
        TOLERANCE = 1e-7
        for s in range(nb_scenarios):
            matrix_A = chance_instance.get_matrix_A(s)
            vector_b = chance_instance.get_vector_b(s)
            for i in range(knap_constraint_qty):
                violation = matrix_A[i, :].dot(var_x_val) - vector_b[i]
                if violation > TOLERANCE:
                    is_scenario_infeasible[s] = True
                    break
        object_scenario_inf = chance_instance._get_scenario_infeasibility(
            var_x_val)
        object_is_scenario_inf = chance_instance._get_infeasible_scenarios(
            object_scenario_inf)
        self.assertEqual(is_scenario_infeasible, object_is_scenario_inf)

    def test_check_infeasibility(self):
        chance_instance = ChanceKnapInstance(self.file_location,
                                             self.continuous_var,
                                             self.epsilon)
        nb_scenarios = 5
        item_qty = 6
        knap_constraint_qty = 10
        var_x_val = np.ones(item_qty)
        var_x_val[[1, 2, 5]] = 0
        var_x_val[3] = 0.6
        is_scenario_feasible = [True for s in range(nb_scenarios)]
        TOLERANCE = 1e-7
        for s in range(nb_scenarios):
            matrix_A = chance_instance.get_matrix_A(s)
            vector_b = chance_instance.get_vector_b(s)
            for i in range(knap_constraint_qty):
                violation = matrix_A[i, :].dot(var_x_val) - vector_b[i]
                if violation > TOLERANCE:
                    is_scenario_feasible[s] = False
                    break
        qty_feas_scenarios = sum(is_scenario_feasible)
        feas_scenarios_needed = math.ceil((1-self.epsilon)*nb_scenarios)
        sol_is_feasible = qty_feas_scenarios >= feas_scenarios_needed
        object_sol_is_feasible = chance_instance.check_feasibility(var_x_val)
        self.assertEqual(sol_is_feasible, object_sol_is_feasible)

    def test_check_feasibility(self):
        chance_instance = ChanceKnapInstance(self.file_location,
                                             self.continuous_var,
                                             self.epsilon)
        nb_scenarios = 5
        item_qty = 6
        knap_constraint_qty = 10
        var_x_val = np.ones(item_qty)
        var_x_val[[1, 2, 5]] = 0
        var_x_val[3] = 0.65
        is_scenario_feasible = [True for s in range(nb_scenarios)]
        TOLERANCE = 1e-7
        for s in range(nb_scenarios):
            matrix_A = chance_instance.get_matrix_A(s)
            vector_b = chance_instance.get_vector_b(s)
            for i in range(knap_constraint_qty):
                violation = matrix_A[i, :].dot(var_x_val) - vector_b[i]
                if violation > TOLERANCE:
                    is_scenario_feasible[s] = False
                    break
        qty_feas_scenarios = sum(is_scenario_feasible)
        feas_scenarios_needed = math.ceil((1-self.epsilon)*nb_scenarios)
        sol_is_feasible = qty_feas_scenarios >= feas_scenarios_needed
        object_sol_is_feasible = chance_instance.check_feasibility(var_x_val)
        self.assertEqual(sol_is_feasible, object_sol_is_feasible)
