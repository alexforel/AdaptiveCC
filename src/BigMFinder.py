# Import packages
import math
import numpy as np
from gurobipy import GRB
from copy import copy

# Import Cython functions
from violations import compute_all_violations
# Import local python functions
from src.optim.BigMKnapsackModel import BigMKnapsackModel
from src.song_big_m import solve_continuous_knapsack
from src.instance.PartitionChanceKnapInstance import \
    PartitionChanceKnapInstance


class BigMFinder(object):
    """
    Find big M parameters of indicator constraints.
    """
    def __init__(self, chance_instance):
        self.chance_instance = chance_instance
        self.nb_scenarios = self.chance_instance.get_nb_scenarios()
        self.song_violations = None
        # Initialize with naive big M
        self.bigM = [self._naive_bigM(s) for s in range(self.nb_scenarios)]

    #   - - - Private methods - - -
    def _naive_bigM(self, s):
        """
        A naive method to initialize the big M, see e.g. Section 3 of:
        Song, Y., Luedtke, J. R., & Küçükyavuz, S. (2014). Chance-constrained
        binary packing problems. INFORMS Journal on Computing, 26(4), 735-747.
        """
        A = self.chance_instance.get_matrix_A(s)
        b = self.chance_instance.get_vector_b(s)
        naive_bigM = np.sum(A, axis=1) - b
        assert len(naive_bigM) == self.chance_instance.get_nb_constraints(s)
        return naive_bigM

    def _single_continuous_belotti_iter(self, s, i, upper_bound):
        """Belotti et al big M tightening method for continuous variables.

        It is solved using a greedy sorting approach.

        Args:
           s (int): scenario index to be tightened
           i (int): constraint index to be tightened
           upper_bound (float): bound used for the knapsack constraint

        Returns:
           new_big_m (float): new big m for constraint (s,i)
        """
        # Get the data from the chance instance
        obj_vector = self.chance_instance.get_matrix_A(s)[i, :]
        obj_constant = self.chance_instance.get_vector_b(s)[i]
        lhs_vector = self.chance_instance.get_vector_c()
        rhs = upper_bound
        # Solve continous single-dimensional knapcksack
        max_violation = solve_continuous_knapsack(
            obj_vector, obj_constant, lhs_vector, rhs)
        return max_violation

    @staticmethod
    def _set_gurobi_params(model, gap, use_one_thread):
        model.grb_model.setParam(GRB.Param.MIPGap, gap)
        model.grb_model.setParam(GRB.Param.IntFeasTol, 1e-9)
        model.grb_model.setParam(GRB.Param.FeasibilityTol, 1e-8)
        if use_one_thread:
            model.grb_model.setParam(GRB.Param.Threads, 1)

    def _single_mip_belotti_iter(self, big_m_knapsack_model, s, i):
        """Belotti et al big M tightening method for binary variables.

        It is solved using a MIP.

        Args:
           s (int): scenario index to be tightened
           i (int): constraint index to be tightened
           upper_bound (float): bound used for the knapsack constraint

        Returns:
           new_big_m (float): new big m for constraint (s,i)
        """
        big_m_knapsack_model.initialize_violation_objective(s, i)
        big_m_knapsack_model.solve()
        return big_m_knapsack_model.get_obj_bnd()

    @staticmethod
    def _print_status(s, nb_scenarios):
        p = math.floor(100*s/nb_scenarios)
        if (p % 10) == 0:
            print('[%d%%] \r' % p, end="")

    def _quantile_big_m(self, violations):
        """Find the quantile of violations."""
        violations.sort()
        epsilon = self.chance_instance.get_epsilon()
        q = int(math.floor(epsilon*self.nb_scenarios)+1)
        return violations[q]

    #   - - - Public methods - - -
    def run_belotti_et_al_big_M(self, vUB,
                                gap=1e-8,
                                use_one_thread=True,
                                verbose=False):
        """
        Applies the big M tightening from:
          Belotti, P., Bonami, P., Fischetti, M., Lodi, A.,
          Monaci, M., Nogales-Gómez, A., & Salvagnin, D. (2016).
          On handling indicator constraints in mixed integer programming.
          Computational Optimization and Applications, 65, 545-566.
        to all scenarios and constraint.

        Args:
           vUB (float): upper bound on the objective of the CCLP model.
        """
        # Check if we have integer variables
        var_type = self.chance_instance.get_var_type()
        integer_vars = (sum(var_type == 0) > 0)

        print('Running Belotti et al for tightening big M\'s.')
        # Integer Vars = run MIP
        if integer_vars:
            # Initialize optimization model
            big_m_knapsack_model = BigMKnapsackModel(self.chance_instance, vUB)
            big_m_knapsack_model.build(verbose=verbose)
            self._set_gurobi_params(big_m_knapsack_model, gap, use_one_thread)
            for s in range(self.nb_scenarios):
                self._print_status(s, self.nb_scenarios)
                nb_constraints = self.chance_instance.get_nb_constraints(s)
                for i in range(nb_constraints):
                    self.bigM[s][i] = self._single_mip_belotti_iter(
                        big_m_knapsack_model, s, i)
        else:
            # Only continuous vars: this is a single-dim knapsack
            for s in range(self.nb_scenarios):
                self._print_status(s, self.nb_scenarios)
                nb_constraints = self.chance_instance.get_nb_constraints(s)
                for i in range(nb_constraints):
                    self.bigM[s][i] = self._single_continuous_belotti_iter(
                        s, i, vUB)

        # Print average big M value
        avgBigM = np.mean([np.mean(self.bigM[s])
                           for s in range(self.nb_scenarios)])
        print('Average bigM value is ', avgBigM)

    def run_song_et_al_big_m(self, chance_instance, partition=None):
        """
        Applies the big M tightening method from:
            Yongjia Song, James R. Luedtke, Simge Küçükyavuz (2014)
            ''Chance-Constrained Binary Packing Problems''
            INFORMS Journal on Computing 26(4):735-747.
            https://doi.org/10.1287/ijoc.2014.0595

        Args:
           chance_instance: a non-partitioned chance instance.
        """
        print('Running Song et al for tightening big M\'s.')
        nb_subsets = self.nb_scenarios
        nb_constraints_per_scenario = chance_instance.get_nb_constraints(0)

        # - Solve (card(S) * card(I))^2 single-dimensional continuous knapsacks
        # Note that this is always calculated over scenarios even if
        # the self.chance_instance is partitioned
        if self.song_violations is None:
            print('Calculating all (s, i, s_prime, i_prime) violations.')
            A_matrices = chance_instance.get_matrices_A()
            b_vectors = chance_instance.get_vectors_b()
            # Call Cython function
            self.song_violations = compute_all_violations(
                A_matrices, b_vectors)

        # - Take quantile of violations over scenarios or subset
        if isinstance(self.chance_instance, PartitionChanceKnapInstance):
            assert partition is not None
        else:
            assert partition is None
        for c in range(nb_subsets):
            nb_constraints = self.chance_instance.get_nb_constraints(c)
            for j in range(nb_constraints):
                if isinstance(self.chance_instance,
                              PartitionChanceKnapInstance):
                    s = j // nb_constraints_per_scenario  # subset index
                    scenario = partition[c][s]
                    constraint = j % nb_constraints_per_scenario
                else:
                    scenario = c
                    constraint = j
                # Get vector of all violations
                all_violations = copy(
                    self.song_violations[scenario, :, constraint])
                if isinstance(self.chance_instance,
                              PartitionChanceKnapInstance):
                    # Aggregate scenarios in subsets if using a partition
                    violations = np.zeros(len(partition))
                    for m, subset in enumerate(partition):
                        violations[m] = min([all_violations[sprime]
                                             for sprime in subset])
                else:
                    violations = all_violations
                # Get big M as quantile of violations
                self.bigM[c][j] = self._quantile_big_m(violations)

        # - Print average big M value
        avgBigM = np.mean([np.mean(self.bigM[s]) for s in range(nb_subsets)])
        print('Average bigM value is ', avgBigM)

    def get_vector_big_M(self, scenario):
        '''
        Returns the big M vector (A^sx<=b^s + M(1-z)) for a specific scenario
        '''
        return self.bigM[scenario]

    def update_big_M(self, nb_scenarios, vUB,
                     method="belotti",
                     chance_instance=None,
                     partition=None):
        """Update the list of big M's for the partition."""
        self.nb_scenarios = nb_scenarios
        # Initialize with naive bigM
        self.bigM = [self._naive_bigM(s) for s in range(self.nb_scenarios)]
        print('Average naive bigM value is ',
              np.mean([np.mean(self.bigM[s])
                       for s in range(self.nb_scenarios)]))

        if method == "belotti":
            assert vUB is not None
            self.run_belotti_et_al_big_M(vUB)
        elif method == 'song':
            assert chance_instance is not None
            self.run_song_et_al_big_m(chance_instance,
                                      partition=partition)
        elif not method == "naive":
            raise ValueError
