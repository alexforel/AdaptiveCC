import numpy as np
import time

from src.TimeManager import TimeManager


class Informer():
    """Parse or query information during the solving process."""

    def __init__(self):
        self.iteration_info = []

    #   - - - Public methods - - -
    def print_iter_info(self, iteration, vUB, vLB, gap):
        current_time = time.time()
        print("\n        ########      -  Iteration "
              + str(iteration) + " -      ########")

        startString = '        ####      '
        print(startString + 'Best vUB: {:06.2f}'.format(vUB))
        if vLB == -np.inf:
            print(startString + "Best vLB:  -inf ")
        else:
            print(startString + 'Best vLB: {:06.2f}'.format(vLB))
        if gap != np.inf:
            print(startString + 'Gap: {:06.2f}'.format(
                gap*100))
        else:
            print(startString + 'Gap: inf')
        print(startString + "Computation Time:  {:06.2f}".format(
            current_time - TimeManager.get_start_time()))

    def adaptive_details(self, split_method, initial_partition_type,
                         projection_method):
        counter_dict = {
            "counter": 1,
            "rescaled_max_violation": 5
            }
        method_details = [counter_dict[split_method],
                          counter_dict[initial_partition_type],
                          counter_dict[projection_method]]
        return method_details

    def store_iteration_info(self, iteration, nb_subsets, vLB, vUB,
                             gap, decimal_places=3):
        """Saves the iteration information."""
        # Preparing decimal places string
        str_decimal_place = "{:."+str(decimal_places)+"f}"
        current_time = time.time()
        start_time = TimeManager.get_start_time()
        self.iteration_info.append(
            [iteration,
             str_decimal_place.format(current_time - start_time),
             str_decimal_place.format(-vLB),
             str_decimal_place.format(-vUB),
             str_decimal_place.format(gap*100),
             nb_subsets])
