import math
import csv
import numpy as np

from src.instance.PartitionChanceKnapInstance import \
    PartitionChanceKnapInstance
from src.Initializer import Initializer
from src.refiner.CostRefiner import CostRefiner
from src.refiner.RandomRefiner import RandomRefiner
from src.UpperBounder import UpperBounder
from src.LowerBounder import LowerBounder
from src.Informer import Informer
from src.Merger import Merger
from src.solver.Solver import Solver


class AdaptivePartitioner(Solver):
    """
    Main class for solving chance-constrained problems
    by iteratively solving reduced problems and
    adapting the partition.
    """
    def __init__(self, chance_instance,
                 split_method='random',
                 initial_partition_type='cost',
                 projection_method='rescaled_max_violation',
                 use_acc_obj=False,
                 time_limit=1800,
                 gap=1e-4):
        super(AdaptivePartitioner, self).__init__(
            chance_instance, time_limit, gap)
        self.xUB = None
        self.zUB = None
        self.did_merge = False
        self.MERGE_TOL = 1.00
        self.split_method = split_method
        self.projection_method = projection_method
        self.initial_partition_type = initial_partition_type
        self.chance_instance_part = PartitionChanceKnapInstance(
            chance_instance)
        self._create_components(use_acc_obj)
        self._setup_initial_partition()

    #   - - - Private methods - - -
    def _init_refiner(self, split_method, use_acc_obj):
        if split_method == 'random':
            refiner = RandomRefiner(self.chance_instance,
                                    use_acc_obj=use_acc_obj)
        elif split_method == 'cost':
            # Cost refiner needs to use accurate obj. split
            assert use_acc_obj
            refiner = CostRefiner(self.chance_instance,
                                  use_acc_obj=use_acc_obj)
        else:
            print('Unknown splitting method, use:')
            print('[\'random\', \'cost\']')
            raise ValueError
        return refiner

    def _create_components(self, use_acc_obj):
        """
        Create all components of the adaptive partitioner.
        """
        self.initializer = Initializer(self.chance_instance,
                                       self.chance_instance_part)
        self.refiner = self._init_refiner(self.split_method, use_acc_obj)
        self.upperbounder = UpperBounder(self.chance_instance_part,
                                         self.split_method,
                                         self.initial_partition_type)
        self.lowerbounder = LowerBounder(self.chance_instance,
                                         self.projection_method)
        self.merger = Merger(self.chance_instance, self.chance_instance_part)
        self.informer = Informer()

    def _setup_initial_partition(self):
        """Create an initial partition and process it.

        (i) Determine the minimum number of subsets, (ii) find an initial
        assignment of scenario to subsets based on self.initial_partition_type,
        and (iii) calculate Ahmed et al.'s quantile bound if scenario costs
        are available.
        """
        # Read instance data and determine minimum number of partitions
        nb_scenarios = self.chance_instance.get_nb_scenarios()
        epsilon = self.chance_instance.get_epsilon()
        self.minimum_partition_size = math.floor(epsilon*nb_scenarios) + 1
        self.nb_subsets = math.floor(nb_scenarios*epsilon) + 1
        # Generate initial partition and load it
        self.partition = self.initializer.create_first_partition(
            self.nb_subsets, partition_type=self.initial_partition_type)
        self.chance_instance_part.load_partition(self.nb_subsets,
                                                 self.partition)
        # Determine quantile bound if possible
        if self.initial_partition_type == "cost":
            self.vUB = self.upperbounder.ahmed_et_al_bound(
                self.initializer.scenario_costs,
                self.chance_instance.get_proba(),
                self.chance_instance.get_epsilon())

    def _improve_vlb_with_candidate_sols(self, candidate_sols, verbose=True):
        """Improve lower bound with candidate solutions."""
        # Check whether one candidate solution can improve current lower bound
        did_improve, xLB, vLB = self.lowerbounder.incumbent_bound(
            candidate_sols, self.vUB, self.vLB, verbose=verbose)
        if did_improve:
            self.xLB = xLB
            self.vLB = vLB

    def _big_M(self, bigMFinder, use_big_M, use_tightening=False):
        """Determine big M parameters for current partition."""
        if use_big_M and (self.nb_subsets > self.minimum_partition_size):
            print("\n - Big M -")
            # Use method in ["belotti", "song", "naive"]
            bigMFinder.update_big_M(self.nb_subsets, self.vUB,
                                    method=self.big_m_method,
                                    chance_instance=self.chance_instance,
                                    partition=self.partition)

    def _upper_bound(self, bigMFinder, use_big_M, use_lazy=False):
        """ Solve upper-bound partitioned problem and check if feasible.

        Args:
            bigMFinder (BigMFinder): to read big M values
            use_big_M (bool): if False, use Gurobi indicator constraint

        Returns:
            bool: True if solution feasible for original problem
        """
        if self.nb_subsets == self.minimum_partition_size:
            # Obtain first upper bound solution by simple sorting
            xUB, z, v_obj, v_bnd = self.upperbounder.first_iteration_bound(
                self.subset_costs, self.subset_sols)
        else:
            # Solve the partitioned problem to obtain an upper bound
            xUB, z, v_obj, v_bnd = self.upperbounder.partition_bound(
                self, self.subset_costs,
                self.vLB, self.zUB, self.merger.deleted_subsets,
                bigMFinder, self._available_time(), use_big_M, use_lazy)

        if self._available_time() <= 0:
            return self.xUB, False

        is_feasible = self.chance_instance.check_feasibility(xUB)

        # - Store results -
        self.zUB = z
        if (v_bnd <= self.vUB + 1e-8):
            if (v_obj <= self.vUB + 1e-8):
                self.xUB = xUB
            self.vUB = v_bnd
        self.lowerbounder.save_feasible_scenarios(
            self.chance_instance.get_feasible_scenarios())
        # Save the iteration info
        self.informer.store_iteration_info(
            self.iteration, self.nb_subsets, self.vLB, self.vUB,
            self._compute_gap())
        return xUB, is_feasible

    def _lower_bound(self, xUB):
        """Solve lower-bound projection step."""
        print('Solving lower-bound deterministic model.')
        self.lowerbounder.vLB = self.vLB
        x, v = self.lowerbounder.deterministic_bound(xUB)
        # Store results
        if v > self.vLB:
            print('Improve lower bound from', self.vLB, 'to', v)
            self.vLB = v
            self.xLB = x
        else:
            print('Could not improve lower bound.')
        # Save the iteration info
        real_gap = self._compute_gap()
        self.informer.store_iteration_info(
            self.iteration, self.nb_subsets, self.vLB, self.vUB,
            real_gap)
        return real_gap

    def _merge_all_feasible_subsets(self, feasible_subsets):
        print("\n - Merge: all feasible subsets with top infeasible subsets -")
        # Find the infeasible subset with largest subset cost
        sorted_subsets = self.merger.sort_subset_costs(
            range(self.nb_subsets), self.subset_costs)
        # Find candidate subsets for merge: all feasible and top infeasible
        for s in sorted_subsets:
            if s not in feasible_subsets:
                top_infeasible_subset = s
                break
        candidate_subsets = feasible_subsets + [top_infeasible_subset]
        # Merge all candidate subsets into one
        self.partition, self.nb_subsets = self.merger.merge_all(
            self.partition, candidate_subsets)
        self._post_merge_processing()
        assert self.nb_subsets == self.minimum_partition_size
        self.did_merge = True

    def _merge_mu_plus_one_feasible(self, feasible_subsets, xUB):
        print("\n - Merge: mu+1 feasible subsets - ")
        # Merge top mu feasible subsets
        self.partition, self.nb_subsets = self.merger.merge_feasible(
            self.partition, xUB, feasible_subsets, self.refiner.mu)
        self._post_merge_processing()
        self.did_merge = True

    def _post_merge_processing(self):
        # Store results and load partition
        self.chance_instance_part.load_partition(
            self.nb_subsets, self.partition)
        # Forget cost of deleted subsets
        sorted_subsets = np.sort(self.merger.deleted_subsets)[::-1]
        for c in sorted_subsets:
            del self.subset_costs[c]
            del self.subset_sols[c]
        # Evaluate cost of new subset
        sols = []
        for c1 in self.merger.target_subsets:
            self.subset_costs[c1], sol = self.evaluator.subset_cost(
                self.partition[c1])
            self.subset_sols[c1] = sol
            sols.append(sol)
            print('New cost of subset ', c1,
                  ' with scenarios', self.partition[c1],
                  ' is ', self.subset_costs[c1])
        # Improve bound with candidate solutions
        self._improve_vlb_with_candidate_sols(sols)

    def _merge(self, xUB, new_subset_costs, use_merger):
        self.did_merge = False
        if use_merger:
            if max(new_subset_costs) <= self.vUB:
                feasible_subsets = self.refiner.feasible_subsets
                nb_feasible = len(feasible_subsets)
                if nb_feasible == self.refiner.mu:
                    # - Merge all feasible subsets with one infeasible -
                    self._merge_all_feasible_subsets(feasible_subsets)
                elif nb_feasible >= (self.refiner.mu + 1):
                    # - Merge mu+1 feasible subsets -
                    self._merge_mu_plus_one_feasible(feasible_subsets, xUB)
                else:
                    raise ValueError

    def _split(self, xUB):
        print("\n - Split - ")
        self.refiner.vUB = self.vUB
        self.refiner.subset_costs = self.subset_costs
        # Compute a new partition from partition refiner
        self.nb_subsets, self.partition = self.refiner.refine(self.partition,
                                                              xUB)
        self.chance_instance_part.load_partition(
            self.nb_subsets, self.partition)
        # Evaluate cost of new subsets
        print('Evaluating the cost of new subsets.')
        new_subset_sols = []
        new_subset_costs = []
        for c in self.refiner.new_subsets:
            subset = self.partition[c]
            cost, sol = self.evaluator.subset_cost(subset)
            new_subset_sols.append(sol)
            new_subset_costs.append(cost)
            if c < len(self.subset_costs):
                self.subset_costs[c] = cost
                self.subset_sols[c] = sol
            else:
                self.subset_costs.append(cost)
                self.subset_sols.append(sol)
        for c in self.refiner.new_subsets:
            print(' Subset ', c, ' with scenarios: ', self.partition[c])
            print('        has cost: ', self.subset_costs[c])
        # Improve bound with candidate solutions
        self._improve_vlb_with_candidate_sols(new_subset_sols)
        return new_subset_costs

    #   - - - Public methods - - -
    def solve(self, bigMFinder, use_big_M=False, big_m_method="belotti",
              use_merger=False, use_lazy=False):
        self.use_merger = use_merger
        self.use_big_M = use_big_M
        self.big_m_method = big_m_method

        #   - Subset cost -
        print('\n Calculating cost of each subset in the initial partition.')
        self.subset_costs, self.subset_sols = self.evaluator.partition_cost(
            self.partition)
        # Improve bound with candidate solutions
        self._improve_vlb_with_candidate_sols(self.subset_sols)

        #  - - - Main loop - - -
        self.iteration = 1
        real_gap = 1
        real_time_left = self._available_time()
        is_feasible = False
        while ((not is_feasible) and (real_time_left > 0)):
            # Print iter info
            self.informer.print_iter_info(
                self.iteration, self.vUB, self.vLB,
                self._compute_gap())

            self._big_M(bigMFinder, use_big_M)

            print("\n - Upper bound - ")
            xUB, is_feasible = self._upper_bound(bigMFinder, use_big_M,
                                                 use_lazy=use_lazy)
            if is_feasible:
                break
            if self._available_time() <= 0:
                print('Interrupting adaptive partitioner: '
                      ' time limit reached.')
                break

            print("\n - Lower bound - ")
            real_gap = self._lower_bound(xUB)
            if real_gap <= self.gap:
                break
            if self._available_time() <= 0:
                print('Interrupting adaptive partitioner: '
                      ' time limit reached.')
                break

            # - Split and merge -
            new_subset_costs = self._split(xUB)
            self._merge(xUB, new_subset_costs, use_merger)
            if self._available_time() <= 0:
                print('Interrupting adaptive partitioner: '
                      ' time limit reached.')
                break

            #   - End of iteration -
            self.iteration += 1
            real_time_left = self._available_time()

        if is_feasible or (real_gap <= self.gap):
            if is_feasible:
                self.vLB = self.vUB
                self.xLB = self.xUB
            print('\n - End - ')
            print('Found optimal solution of CCLP: ', self.xUB)
            print('with objective: ', self.vUB)

    def write_all_computation_details(self, output_file_location,
                                      decimal_places=3):
        # Preparing decimal places string
        str_decimal_place = "{:."+str(decimal_places)+"f}"
        # Extracting computation parameters and saving to list
        computation_parameters = [str_decimal_place.format(self.time_limit),
                                  str_decimal_place.format(self.gap*100),
                                  self.use_big_M*1]
        adaptive_details = self.informer.adaptive_details(
            self.split_method, self.initial_partition_type,
            self.projection_method)

        instance_details, sup_computation_dets = super(
            AdaptivePartitioner, self)._get_computation_details(
                decimal_places=decimal_places)
        size_partition = self.chance_instance_part.get_nb_scenarios()
        computation_details = ([self.iteration, size_partition] +
                               sup_computation_dets)
        written_line = (instance_details + computation_parameters +
                        adaptive_details + computation_details)
        with open(output_file_location, 'w', newline='\n') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(written_line)

    def write_iteration_details(self, output_file_location):
        # Preparing decimal places string
        with open(output_file_location, 'w', newline='\n') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for line in self.informer.iteration_info:
                writer.writerow(line)
