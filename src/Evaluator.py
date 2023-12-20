from src.optim.DeterModel import DeterModel
from math import floor


class Evaluator():
    """
    Class used to evaluate all deterministic models that may
    be used at several places in the code. For instance, it
    can calculate the cost of a scenario, subset, or partition.
    """

    def __init__(self, chance_instance):
        self.chance_instance = chance_instance

    @staticmethod
    def _print_progress(s, nb_s):
        p = floor(100*s/nb_s)
        if (p % 10) == 0:
            print('[%d%%] \r' % p, end="")

    #   - - - Public methods - - -
    def subset_cost(self, subset):
        """Solve the deterministic model that satisfies
        all the scenarios in the given subset.

        Args:
            subset (list[int]): a (list of) scenario(s)

        Returns:
            float: optimal objective
            np.array(float/binary): optimal solution variables
        """
        deterministicModel = DeterModel(self.chance_instance)
        deterministicModel.build(subset)
        deterministicModel.solve()
        subset_cost = deterministicModel.get_obj_val()
        subset_sol = deterministicModel.get_var_x_val()
        return subset_cost, subset_sol

    def scenario_costs(self, scenarios):
        """
        Evaluate the single-scenario cost of
        all the scenarios in the given list.
        """
        subset_costs = []
        for s in scenarios:
            self._print_progress(s, len(scenarios))
            cost, _ = self.subset_cost(s)
            subset_costs.append(cost)
        return subset_costs

    def get_all_single_scenario_costs(self):
        """
        Evaluate the single-scenario cost of
        all the scenarios in the chance instance.
        """
        nb_scenarios = self.chance_instance.get_nb_scenarios()
        scenarios = range(nb_scenarios)
        scenario_costs = self.scenario_costs(scenarios)
        return scenario_costs

    def partition_cost(self, partition):
        """
        Evaluate the subset cost of all the subsets
        in the given partition.

        Returns:
            list[float]: optimal objectives
            list[np.array(float/binary)]: optimal solution variables
        """
        subset_costs = []
        subset_sols = []
        count = 0
        for subset in partition:
            self._print_progress(count, len(partition))
            count += 1
            cost, sol = self.subset_cost(subset)
            subset_costs.append(cost)
            subset_sols.append(sol)
        return subset_costs, subset_sols
