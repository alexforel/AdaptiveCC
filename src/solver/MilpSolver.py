import csv

from src.BigMFinder import BigMFinder
from src.solver.Solver import Solver
from src.UpperBounder import UpperBounder


class MilpSolver(Solver):
    """
    Solve chance-constrained problem in extended formulation,
    i.e., having one binary indicator variable per scenario.
    """
    def __init__(self, chance_instance, time_limit=1800, gap=1e-4):
        super(MilpSolver, self).__init__(chance_instance, time_limit, gap)
        self.big_m_finder = BigMFinder(self.chance_instance)
        self.upper_bounder = UpperBounder(None, None, None)

    #   - - - Private methods - - -
    def _compute_big_m(self, big_m_method="belotti"):
        """Computes the big M's according to the input method string."""
        if big_m_method == "belotti":
            # Compute single-scenario costs
            scenario_costs = self.evaluator.get_all_single_scenario_costs()
            # Find quantile upper bound from Ahmed et al
            vUB = self.upper_bounder.ahmed_et_al_bound(
                scenario_costs,
                self.chance_instance.get_proba(),
                self.chance_instance.get_epsilon())
            # Run Belotti et al big M tightening method
            self.big_m_finder.run_belotti_et_al_big_M(vUB)
        elif big_m_method == "song":
            self.big_m_finder.run_song_et_al_big_m(self.chance_instance)
        elif big_m_method == "naive":
            print('Warning: using naive big M is not recommended.')
            pass
        else:
            raise ValueError("Incorrect big m method provided")
        # Save time needed to obtain big-M parameters
        self.big_m_time = (self.time_limit - self._available_time())

    def _save_computation_parameters(self, use_big_m, big_m_method):
        self.use_big_M = use_big_m*1 + (not use_big_m)*0
        self.big_m_method = (1*(big_m_method == "naive")
                             + 2*(big_m_method == "ahmed_belotti")
                             + 3*(big_m_method == "qiu_et_al"))

    #   - - - Public methods - - -
    def solve(self, use_big_m=True, big_m_method="naive",
              save_bounds=False, path=None):
        """Solves extended CCLP model with given params."""
        # Compute big M's according to big_m_method
        if use_big_m:
            self._compute_big_m(big_m_method=big_m_method)

        x, z, v_obj, v_bnd = self.solve_cclp_model(
            self.chance_instance, self.big_m_finder,
            time_limit=self._available_time(),
            elapsed_time=(self.time_limit - self._available_time()),
            gap=self.gap,
            use_big_M=use_big_m, save_bounds=save_bounds,
            path=path, verbose=True)
        self._save_computation_parameters(use_big_m, big_m_method)
        self.xLB = x
        self.zLB = z
        self.vLB = v_obj
        self.vUB = v_bnd

    def write_all_computation_details(self, output_file_location,
                                      decimal_places=3):
        # Preparing decimal places string
        str_decimal_place = "{:."+str(decimal_places)+"f}"
        # Extracting computation parameters and saving to list
        computation_parameters = [str_decimal_place.format(self.time_limit),
                                  str_decimal_place.format(self.gap*100),
                                  self.use_big_M,
                                  self.big_m_method]
        instance_details, computation_details = self._get_computation_details()
        written_line = (instance_details + computation_parameters
                        + [self.big_m_time] + computation_details)
        with open(output_file_location, 'w', newline='\n') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(written_line)
