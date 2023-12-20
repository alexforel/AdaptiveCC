import random
import numpy as np

from src.Evaluator import Evaluator


class Initializer(object):
    """Create initial scenario partition."""
    def __init__(self, chance_instance, chance_instance_part):
        self.chance_instance = chance_instance
        self.chance_instance_part = chance_instance_part
        self.nb_scenarios = chance_instance.get_nb_scenarios()
        self.evaluator = Evaluator(self.chance_instance)

    #   - - - Private methods - - -
    @staticmethod
    def _random_partition(scenarios, n):
        """
        Randomly assigns elements of scenarios into n partitions of
        approximately equal size.
        From: https://stackoverflow.com/questions/3352737/
        """
        scenarios = random.sample(scenarios, len(scenarios))
        return [scenarios[i::n] for i in range(n)]

    def _single_scenario_cost_partition(self, scenarios, nb_clusters):
        """
        Partition based on the costs of the single-scenario problems.

        For all scenarios, solve independently all the CCLP for a
        single-scenario, i.e., deterministic models; collect the costs;
        and partition the scenarios so that the spread of the costs
        is maximized over the partitions.
        """
        print('Solving single-scenario problems: ')
        scenario_costs = self.evaluator.get_all_single_scenario_costs()
        scenario_costs = np.array(scenario_costs)
        # Sort costs
        sorted_indices = np.argsort(-scenario_costs)
        # Partition based on costs
        partitions = []
        for i in range(nb_clusters):
            partition = []
            for j in range(i, self.nb_scenarios, nb_clusters):
                partition.append(scenarios[sorted_indices[j]])
            partitions.append(partition)
        return partitions, scenario_costs

    def _check_partitions(self, partitions, nb_clusters):
        """
        Check that all scenarios are assigned and there is no duplicate.
        """
        all_scenarios = [item for sublist in partitions for item in sublist]
        assert (len(all_scenarios) == self.nb_scenarios)
        assert (np.unique(all_scenarios).size == self.nb_scenarios)
        assert (len(partitions) == nb_clusters)

    #   - - - Public methods - - -
    def create_first_partition(self, nb_clusters, partition_type='random'):
        assert (nb_clusters <= self.nb_scenarios)
        # Create list of scenario indices
        scenarios = list(range(self.nb_scenarios))
        print(' - Initialize procedure with first partition -')
        # Create initial partition depending on type
        if partition_type == 'random':
            partitions = self._random_partition(
                scenarios, nb_clusters)
        elif partition_type == 'cost':
            partitions, scenario_costs = self._single_scenario_cost_partition(
                scenarios, nb_clusters)
            self.scenario_costs = scenario_costs
        else:
            raise NotImplementedError
        # Check results and return partitions
        self._check_partitions(partitions, nb_clusters)
        return partitions
