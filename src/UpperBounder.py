import numpy as np

from src.Warmstarter import Warmstarter


class UpperBounder():
    """
    Class that contains all methods for deriving upper bounds:
    e.g., the quantile bound of Ahmed et al, or the upper bound
    obtained from solving a partitioned problem.
    """

    def __init__(self, chance_instance_part, split_method,
                 initial_partition_type):
        self.chance_instance_part = chance_instance_part
        self.split_method = split_method
        self.initial_partition_type = initial_partition_type
        self.warmstarter = Warmstarter()

    #   - - - Private methods - - -
    def _check_results(self, x, v_obj, adaptivePartitioner):
        if self.chance_instance_part.continuous_var:
            if ((not adaptivePartitioner.did_merge)
               or (adaptivePartitioner.did_merge
                   and adaptivePartitioner.MERGE_TOL > 1.00)):
                try:
                    assert not np.allclose(x, adaptivePartitioner.xUB)
                except AssertionError:
                    print('Error: found same solution in two successive'
                          ' iterations of the upper bound problem.')
                    print('Solution xUB: ', x)
                    raise
            try:
                assert v_obj <= (adaptivePartitioner.MERGE_TOL
                                 * adaptivePartitioner.vUB + 1e-8)
            except AssertionError:
                print('Error: vUB increased in this iteration!')
                print('From ', adaptivePartitioner.vUB, ' to ', v_obj)
                raise

    #   - - - Public methods - - -
    def ahmed_et_al_bound(self, scenario_costs, proba, epsilon) -> float:
        """
        Compute the quantile bound provided in:
        Ahmed, S., Luedtke, J., Song, Y., & Xie, W. (2017). Nonanticipative
        duality, relaxations, and formulations for chance-constrained=
        stochastic programs. Mathematical Programming, 162, 51-81.

        The bound is the epsilon-th largest cost in the given
        list of scenario costs
        """
        sorted_scenarios = np.argsort(scenario_costs)
        index_counter = 0
        total_proba = 0
        for s in sorted_scenarios:
            index_counter += 1
            scenario_proba = proba[s]
            total_proba += scenario_proba
            if total_proba >= epsilon + 1e-7:
                bound = scenario_costs[s]
                print("Quantile upper bound is: " + str(bound))
                self.bounding_scenario = s
                return bound

    def first_iteration_bound(self, subset_costs, subset_sols):
        """
        Find the subset with maximum cost and return its single-subset
        optimal solution.
        """
        max_subset = np.argmax(subset_costs)
        x = subset_sols[max_subset]
        z = np.zeros(len(subset_costs))
        z[max_subset] = 1
        v_obj = subset_costs[max_subset]
        v_bnd = subset_costs[max_subset]
        print('Upper bound is maximum subset cost', v_obj,
              ' given by subset ', max_subset)
        return x, z, v_obj, v_bnd

    def partition_bound(self, adaptivePartitioner, subset_costs,
                        vLB, zUB, deleted_subsets, bigMFinder,
                        real_time_left, use_big_M, use_lazy):
        """
        Get upper bound by solving partitioned chance-constrained
        problem.

        Args:
            adaptivePartitioner (AdaptivePartitioner)
            subset_costs (list[float]): all single-subset costs
            vLB (float): current best lower bound
            zUB (dict): indicator variables solutions from last
                        upper bound problem
            deleted_subsets (list[int]): list of subsets deleted in
                                         merging step of last iter
            bigMFinder (BigMFinder): provider for big M values
            real_time_left (float): in seconds
            use_big_M (bool): whether to use big M or Gurobi indicator
                              constrainst

        Returns:
            np.array(float/binary): optimal solution
            np.array(binary): optimal solution of indicators
            float: optimal objective
            float: upper bound on objective (if model not
                   solved to optimality)
        """
        # Warm-start the indicator variables of the subsets
        z_start = self.warmstarter.get_z_start(zUB, deleted_subsets)

        # Pruning: find subsets whose indicator can be fixed to 0
        TOL = 1e-3
        prune_indices = np.where(np.array(subset_costs) <= (vLB - TOL))[0]

        # Solve lower-bound partitioned problem and get solution and cost
        x, z, v_obj, v_bnd = adaptivePartitioner.solve_cclp_model(
            self.chance_instance_part, bigMFinder,
            z_start=z_start,
            prune_indices=prune_indices,
            time_limit=real_time_left,
            gap=1e-8,
            use_big_M=use_big_M,
            use_lazy=use_lazy,
            verbose=True)

        # - Sanity check -
        self._check_results(x, v_obj, adaptivePartitioner)
        return x, z, v_obj, v_bnd
