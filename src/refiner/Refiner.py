import math
from copy import deepcopy

from src.TimeManager import TimeManager


class Refiner():
    """Refiner class that split subsets."""

    def __init__(self, chance_instance, use_acc_obj):
        self.chance_instance = chance_instance
        self.time_manager = TimeManager()
        self.use_acc_obj = use_acc_obj

    #   - - - Private methods - - -
    def _initialize_refinement(self, partition, x_UB):
        """Store information before refining.

        Args:
            partition (list[list[int]]): partition
            x_UB: solution of last partitioned UB problem.
        """
        self.x_UB = x_UB  # Store best upper bound solution
        self.splitted_subsets = []  # Store subsets that are splitted
        self.new_subsets = []  # Store new subsets
        self.old_partition = partition
        self.partition = deepcopy(partition)

    def _read_infeasible_scenarios_from_chance_instance(self):
        """Read infeasibility data from chance_instance."""
        ci = self.chance_instance
        # List of boolean: True if scenario is infeasible
        self.is_scenario_infeasible = ci.is_scenario_infeasible
        # Indices of infeasible scenarios
        self.infeasible_scenarios = ci.infeasible_scenarios
        # Indices of feasible scenarios
        self.max_infeasibility_scenarios = ci.max_infeasibility_scenarios

    @staticmethod
    def _count_infeasible_scenarios(partition: list[list[int]],
                                    infeasible_scenarios: list[int]):
        """
        Count how many scenarios are infeasible in each subset.
        Args:
            infeasible_scenarios[list]: list of all infeasible scenarios
        Returns:
            list[int]: number of infeasible scenarios in each subset
        """
        nb_subsets = len(partition)
        nb_inf_scenarios = [0] * nb_subsets
        assert isinstance(infeasible_scenarios, list)
        # If the list is empty, all scenarios are feasible
        if len(infeasible_scenarios) == 0:
            return nb_inf_scenarios
        # Otherwise, count the nb of infeasible scenarios in each subset
        for c in range(nb_subsets):
            for s in partition[c]:
                if s in infeasible_scenarios:
                    nb_inf_scenarios[c] += 1
        return nb_inf_scenarios

    def _find_infeasible_subsets(self, partition):
        """Returns the list of feasible and infeasible subsets."""
        # Count how many scenarios are infeasible in each subset
        self.nb_inf_scenarios = self._count_infeasible_scenarios(
            partition, self.infeasible_scenarios)
        # Find feasible and infeasible subsets
        feasible_subsets = []
        infeasible_subsets = []
        for c in range(len(partition)):
            if self.nb_inf_scenarios[c] == 0:
                feasible_subsets.append(c)
            else:
                infeasible_subsets.append(c)
        return feasible_subsets, infeasible_subsets

    def _find_number_of_splits_to_perform(self, nb_infeasible_subsets):
        """Find number of splits to exclude last-found xUB.

        Args:
            nb_infeasible_subsets (int): nb of subsets that are infeasible.

        Returns:
            int: number of splits to perform
        """
        epsilon = self.chance_instance.get_epsilon()
        nb_scenarios = self.chance_instance.get_nb_scenarios()
        nb_subset_tolerance = int(math.floor(epsilon*nb_scenarios))
        # Determine how many splits need to be performed
        mu = nb_subset_tolerance + 1 - nb_infeasible_subsets
        try:
            assert (mu >= 1)
        except AssertionError:
            print('Error when calculating number of splits to perform.')
            print('The number of infeasible clusters:', nb_infeasible_subsets)
            print('should be smaller or equal to:', nb_subset_tolerance)
            raise
        return mu

    @staticmethod
    def _is_splittable(nb_inf_scenarios_in_subset):
        """
        A subset can only be split if it has at
        least two infeasible scenarios.
        """
        return nb_inf_scenarios_in_subset >= 2

    @staticmethod
    def _split_sorted_subset(sorted_scenarios):
        """
        Split an ordered subset in two and assign sequentially
        each element to a subset.

        Args:
            sorted_scenarios (list[int]): list of sorted scenarios
        """
        nb_scenarios = len(sorted_scenarios)
        assert nb_scenarios >= 2
        # Assign scenarios to subsets based on ordering
        even_subset = sorted_scenarios[0:nb_scenarios:2]
        odd_subset = sorted_scenarios[1:nb_scenarios:2]
        return even_subset, odd_subset

    def _split_scenarios(self, sorted_scenarios):
        if len(sorted_scenarios) >= 2:
            left_subset, right_subset = self._split_sorted_subset(
                sorted_scenarios)
        else:
            left_subset = sorted_scenarios
            right_subset = []
        return left_subset, right_subset

    def _split_scenarios_subset(self, scenarios, c):
        """Split scenarios in two sets according to indices."""
        assert len(self.partition[c]) == len(scenarios)
        assert set(self.partition[c]) == set(scenarios)
        # Split scenarios into infeasible and feasible lists
        infeasScenarios = [s for s in scenarios
                           if self.is_scenario_infeasible[s]]
        feasScenarios = [s for s in scenarios
                         if not self.is_scenario_infeasible[s]]
        # Assign scenarios to subsets
        assert len(infeasScenarios) >= 2
        left_infeas_subset, right_infeas_subset = self._split_scenarios(
            infeasScenarios)
        left_feas_subset, right_feas_subset = self._split_scenarios(
            feasScenarios)
        # Split subset c and add new subset to the end of partition
        self.partition[c] = left_infeas_subset + left_feas_subset
        self.partition.append(right_infeas_subset + right_feas_subset)

    def _traverse_subsets_and_split(self, sorted_subsets):
        nb_inf_scenarios = self._count_infeasible_scenarios(
            self.partition, self.infeasible_scenarios)
        for c in sorted_subsets:
            if (self.mu_counter < self.mu) and (nb_inf_scenarios[c] >= 2):
                print(' -> subset ', c)
                self.mu_counter += 1
                # Get scenarios ordered according to criterion
                scenarios = self._get_sorted_scenarios(self.partition[c])
                # Split scenarios in two sets according to their indices
                self._split_scenarios_subset(scenarios, c)
                # Store subsets that have been splitted and created
                if c not in self.splitted_subsets:
                    self.splitted_subsets.append(c)
                if c not in self.new_subsets:
                    self.new_subsets.append(c)
                self.new_subsets.append(len(self.partition)-1)

    def _split_top_mu_subset(self):
        """Loop over all subsets until mu of them have been split."""
        self.mu_counter = 0
        if self.use_acc_obj:
            sorted_subsets = self._get_sorted_subsets(self.partition)
            self._evaluate_accurate_splits(sorted_subsets)
        print('Splitting subsets:')
        while self.mu_counter < self.mu:
            if self.use_acc_obj:
                self._perform_accurate_split()
            else:
                sorted_subsets = self._get_sorted_subsets(self.partition)
                self._traverse_subsets_and_split(sorted_subsets)
        assert self.mu_counter == self.mu

    def _get_sorted_scenarios(self, scenarios):
        """
        Return the input scenarios in the order corresponding to
        the method's criterion (e.g., max violation or cost increase).
        """
        raise NotImplementedError

    def _get_sorted_subsets(self, partition):
        """
        Return the subsets in the order corresponding to
        the method's criterion (e.g., max violation or cost increase).
        """
        raise NotImplementedError

    #   - - - Check and asserts - - -
    @staticmethod
    def _assert_enough_splits(muCounter, mu):
        assert (muCounter >= mu)

    @staticmethod
    def _assert_no_subset_is_empty(partition):
        for c in partition:
            assert (not c == [])

    def _assert_nb_subsets_equal_target(self, partition, old_partition, mu):
        assert (len(partition) == len(old_partition) + mu)

    def _assert_nb_subsets_increased(self, partition, old_partition):
        assert (len(partition) > len(old_partition))

    def _assert_all_scenarios_used_once(self, partition):
        allScenarios = [s for c in partition for s in c]
        trueAllScenarios = list(range(self.chance_instance.get_nb_scenarios()))
        assert set(allScenarios) == set(trueAllScenarios)
        assert len(allScenarios) == len(trueAllScenarios)

    def _check_final_partition(self):
        assert not (self.partition == self.old_partition)
        # self._assert_enough_splits(self.mu_counter, self.mu)
        self._assert_no_subset_is_empty(self.partition)
        self._assert_no_subset_is_empty(self.old_partition)
        self._assert_nb_subsets_equal_target(
            self.partition, self.old_partition, self.mu)
        self._assert_all_scenarios_used_once(self.partition)
        self._assert_nb_subsets_increased(
            self.partition, self.old_partition)

    #   - - - Public methods - - -
    def refine(self, partition, x_UB):
        # - Pre-processing -
        self._initialize_refinement(partition, x_UB)
        # Read infeasible scenarios
        self._read_infeasible_scenarios_from_chance_instance()
        # Find mu: the number of splits to perform
        self.feasible_subsets, inf_subsets = self._find_infeasible_subsets(
            self.partition)
        self.mu = self._find_number_of_splits_to_perform(len(inf_subsets))

        # - Refine -
        print('Splitting ', self.mu, ' subsets:')
        self._split_top_mu_subset()

        # - Check output -
        self._check_final_partition()
        self.feasible_subsets, self.infeasible_subsets = (
            self._find_infeasible_subsets(self.partition))
        return len(self.partition), self.partition
