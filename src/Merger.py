import numpy as np


class Merger():
    """Merge subsets."""
    def __init__(self, chance_instance, chance_instance_part):
        self.chance_instance = chance_instance
        self.chance_instance_part = chance_instance_part
        self.deleted_subsets = []

    #   - - - Private methods - - -
    def _single_subset_margin(self, c, xUB):
        """
        Measure the constraint satisfaction margin
        for each subset/constraint.
        """
        nb_constraints = self.chance_instance_part.get_nb_constraints(c)
        A = self.chance_instance_part.get_matrix_A(c)
        b = self.chance_instance_part.get_vector_b(c)
        subset_margin = np.inf
        for i in range(nb_constraints):
            temp_margin = b[i] - A[i, :].dot(xUB)
            # Rescale margin
            rescaling = np.linalg.norm(A[i, :])
            if rescaling != 0:
                temp_margin = temp_margin/rescaling
            # Find minimum margin over all constraints
            subset_margin = min(temp_margin, subset_margin)
        return subset_margin

    def _delete_subsets(self, partition):
        """Delete the right subsets from list of subsets"""
        sorted_subsets = np.sort(self.deleted_subsets)[::-1]
        for c in sorted_subsets:
            del partition[c]
        return partition

    def _merge_two_subsets_in_partition(self, partition, c1, c2):
        targetSubset = min(c1, c2)
        deletedSubset = max(c1, c2)
        # Keep in memory the target subset
        if targetSubset not in self.target_subsets:
            self.target_subsets.append(targetSubset)
        # Merge by summing the two lists of scenarios
        partition[targetSubset] = (partition[targetSubset]
                                   + partition[deletedSubset])
        # Keep in memory the deleted subset
        self.deleted_subsets.append(deletedSubset)
        print(' -> merged subset: ', deletedSubset, ' into ', targetSubset)
        return partition

    #   - - - Public methods - - -
    @staticmethod
    def sort_subset_costs(candidates, subset_costs):
        return [c for _, c in sorted(
            zip(subset_costs, candidates), reverse=True)]

    @staticmethod
    def can_merge_feasible(new_subset_costs, vUB, nb_feasible, merge_tol):
        """Check if conditions to merge two feasible subsets are met.

        The condition is that at least two subsets are currently feasible
        and the maximum cost of the new subsets that have been split is
        smaller than (a tolerance around) the best known vUB.

        Args:
            new_subset_costs (list[float]): list of costs of new subsets
            vUB (float): best known upper bound
            nb_feasible (int): nb feasible subsets in last upper bound iter
            merge_tol (float): tolerance around merge

        Returns:
            bool: conditions are met -> merge
        """
        assert merge_tol >= 1.0
        return (max(new_subset_costs) <= merge_tol*vUB and (nb_feasible >= 2))

    def find_inf_candidate(self, infeasible_subsets, subset_costs, vUB):
        # Select infeasible subsets with cost smaller than vUB
        candidate_infeasible = []
        for c in infeasible_subsets:
            if subset_costs[c] <= vUB:
                candidate_infeasible.append(c)
        print('Candidate infeasible subsets:', candidate_infeasible)
        # Sort candidate subsets in decreasing order of cost
        top_candidate_infeasible = self.sort_subset_costs(
            candidate_infeasible, subset_costs)
        print('Sorted in decreasing order of subset cost:',
              top_candidate_infeasible)
        return top_candidate_infeasible

    def merge_all(self, partition, candidate_subsets):
        init_size = len(partition)
        self.target_subsets = []
        self.deleted_subsets = []
        # Merge all candidate subsets together
        min_subset = min(candidate_subsets)
        for s in candidate_subsets:
            if not s == min_subset:
                partition = self._merge_two_subsets_in_partition(
                    partition, min_subset, s)
        # Delete all subsets that have been merged
        partition = self._delete_subsets(partition)
        # - Sanity check -
        assert (len(partition)+len(self.deleted_subsets) == init_size)
        return partition, len(partition)

    def merge(self, partition, top_candidate_subsets,
              top_candidate_infeasible):
        init_size = len(partition)
        self.target_subsets = []
        self.deleted_subsets = []
        # Merge all candidate new subsets
        for i, c in enumerate(top_candidate_subsets):
            partition = self._merge_two_subsets_in_partition(
                partition, top_candidate_infeasible[i], c)
        # Delete all subsets that have been merged
        partition = self._delete_subsets(partition)
        # - Sanity check -
        assert (len(partition)+len(self.deleted_subsets) == init_size)
        return partition, len(partition)

    def merge_feasible(self, partition, xUB, candidate_subsets, mu):
        """
        Sort the subsets in decreasing order of their satisfaction
        and merge the top two subsets.
        """
        init_size = len(partition)
        self.target_subsets = []
        self.deleted_subsets = []
        # Calculate margin to feasible region of each subset
        margins = dict()
        for c in candidate_subsets:
            margins[c] = self._single_subset_margin(c, xUB)
        # Sort subsets in decreasing orders of their margins
        top_subsets = sorted(margins, key=margins.get, reverse=True)[:(mu+1)]
        assert len(top_subsets) == (mu+1)

        # - Merge -
        print('Candidate subsets: ', candidate_subsets)
        print('Merging mu subsets with maximum margin to a constraint.')
        partition, nb_subsets = self.merge_all(partition, top_subsets)

        # - Sanity check -
        assert (len(partition)+len(self.deleted_subsets) == init_size)
        return partition, nb_subsets
