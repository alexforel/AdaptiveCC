import numpy as np
import math

from src.optim.AccObjModel import AccObjModel
from src.refiner.Refiner import Refiner


class CostRefiner(Refiner):
    """Refiner class that splits subsets according to their subset costs."""

    def __init__(self, chance_instance, use_acc_obj=False):
        super().__init__(chance_instance, use_acc_obj)
        self.memory_costs = dict()
        self.memory_left_subset = dict()
        self.memory_right_subset = dict()

    #   - - - Private methods - - -
    def _get_sorted_scenarios(self, scenarios):
        # Not implemented: use accurate obj increase
        raise NotImplementedError

    def _get_sorted_subsets(self, current_part):
        """Sort subsets in decreasing order of their infeasibility."""
        assert len(self.subset_costs) == len(self.partition)
        index_min_subset_cost = np.argsort(
            np.array(self.subset_costs)).tolist()
        return index_min_subset_cost

    def _evaluate_single_accurate_split(self, c):
        # Read scenarios and infeasible scenarios in input subset
        scenarios = self.partition[c]
        infeasible_scenarios = [s for s in scenarios
                                if s in self.infeasible_scenarios]
        # Get key of dictionary
        key = (frozenset(scenarios), frozenset(infeasible_scenarios))
        if key not in self.memory_costs:
            # Solve accurate obj model
            post_split_costs, left_subsets, right_subsets = (
                    self.accurate_obj_split(scenarios, infeasible_scenarios))
            self.post_split_costs[c] = post_split_costs
            self.left_subsets[c] = left_subsets
            self.right_subsets[c] = right_subsets
            self.count += 1
            # Store solution in memory
            self.memory_costs[key] = post_split_costs
            self.memory_left_subset[key] = left_subsets
            self.memory_right_subset[key] = right_subsets
        else:
            # Read solution of accurate obj from memory
            self.post_split_costs[c] = self.memory_costs[key]
            self.left_subsets[c] = self.memory_left_subset[key]
            self.right_subsets[c] = self.memory_right_subset[key]

    def _evaluate_accurate_splits(self, sorted_subsets, nb_top=None):
        if nb_top is None:
            # nb_top = max(int(math.floor(len(sorted_subsets)/5)), self.mu)
            nb_top = len(sorted_subsets)
        nb_inf_scenarios = self._count_infeasible_scenarios(
            self.partition, self.infeasible_scenarios)
        # Identify sorted subsets that have at least two infeasible scenarios
        candidate_subsets = [c for c in sorted_subsets
                             if nb_inf_scenarios[c] >= 2]
        # Evaluate max cost of top candidate subsets to split
        nb_candidates = min(len(candidate_subsets), nb_top)
        print('Evaluate accurate obj when splitting top', nb_candidates,
              'candidate subsets.')
        self.post_split_costs = dict()
        self.left_subsets = dict()
        self.right_subsets = dict()
        self.count = 0
        for i in range(nb_candidates):
            p = math.floor(100*i/nb_candidates)
            if (p % 10) == 0:
                print('[%d%%] \r' % p, end="")
            self._evaluate_single_accurate_split(candidate_subsets[i])
        print('Solved', self.count, 'new accurate obj models and '
              'reused previous solutions for',
              nb_candidates-self.count, 'others.')

    def _perform_accurate_split(self):
        # - Perfom split -
        # Find the ``best`` candidate subset to split
        free_mergers = [c for c, v in self.post_split_costs.items()
                        if v <= self.vUB]
        if len(free_mergers) > 0:
            split_c = -1
            v = -1e6
            for c in free_mergers:
                if self.post_split_costs[c] >= v:
                    split_c = c
                    v = self.post_split_costs[c]
        else:
            split_c = min(self.post_split_costs, key=self.post_split_costs.get)
        print(' -> subset ', split_c)
        # Update partition with split
        self.partition[split_c] = self.left_subsets[split_c]
        self.partition.append(self.right_subsets[split_c])
        # Evaluate new subsets
        nb_inf_scenarios = self._count_infeasible_scenarios(
            self.partition, self.infeasible_scenarios)
        if nb_inf_scenarios[split_c] >= 2:
            self._evaluate_single_accurate_split(split_c)
        else:
            self.post_split_costs.pop(split_c)
        new_c = len(self.partition)-1
        if nb_inf_scenarios[new_c] >= 2:
            self._evaluate_single_accurate_split(new_c)
        # Update counter
        self.mu_counter += 1
        # Store new or modified subsets
        if split_c not in self.splitted_subsets:
            self.splitted_subsets.append(split_c)
        if split_c not in self.new_subsets:
            self.new_subsets.append(split_c)
        self.new_subsets.append(len(self.partition)-1)

    #   - - - Public methods - - -
    def accurate_obj_split(self, scenarios, infeasible_scenarios):
        # - Solve optimization model -
        model = AccObjModel(self.chance_instance, scenarios,
                            infeasible_scenarios)
        model.build()
        model.solve()
        max_cost = model.get_obj_val()
        left_subset, right_subset = model.get_subsets()
        return max_cost, left_subset, right_subset