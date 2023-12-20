import numpy as np

from src.instance.ChanceKnapInstance import ChanceKnapInstance


class PartitionChanceKnapInstance(ChanceKnapInstance):
    """Partitioned instance of a MKnap CCLP."""

    def __init__(self, source=None):
        if source is not None:
            self.__dict__.update(source.__dict__)

    #   - - - Public methods - - -
    def load_partition(self, nb_subsets, partition):
        """Setup instance with the input partition."""
        self.nb_subsets = nb_subsets
        self.partition = partition
        # Read the number of constraints: the sum over all the scenarios'
        self.nb_constraints_part = np.array(
            [self.nb_constraints[partition[subset]].sum()
             for subset in range(nb_subsets)])
        # Read probability of each subset: the min of its scenarios' proba
        self.proba_part = np.array(
            [self.proba[partition[subset]].min()
             for subset in range(nb_subsets)])
        # Normalize probability of all subsets to sum to 1
        total_proba = np.sum(self.proba_part)
        self.proba_part = self.proba_part/total_proba
        self.epsilon_part = self.epsilon/total_proba

    def get_matrix_A(self, subset):
        """
        Returns the left hand side A^s matrix (A^sx<=b) for
        a given subset index.
        """
        nb_constraints_set = self.nb_constraints_part[subset]
        return self.matrices_A[self.partition[subset],
                               :, :].reshape((nb_constraints_set,
                                              self.nb_vars))

    def get_vector_b(self, subset):
        """
        Returns the right hand side b^s vector (A^sx<=b^s)
        for a given subset index.
        """
        nb_constraints_set = self.nb_constraints_part[subset]
        return self.vectors_b[self.partition[subset],
                              :].reshape((nb_constraints_set, ))

    def get_nb_scenarios(self):
        """
        Returns the number of subsets in the current partition.
        """
        return self.nb_subsets

    def get_nb_constraints(self, subset):
        """
        Returns the number of constraints of the given subset index.
        """
        return self.nb_constraints_part[subset]

    def get_proba(self):
        """
        Returns the probability p_s of all subsets.
        """
        return self.proba_part

    def get_epsilon(self):
        """
        Returns the chance-constraint tolerance parameter.
        """
        return self.epsilon_part
