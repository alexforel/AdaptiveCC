import random
import sys
import os
import signal

from src.instance.ChanceKnapInstance import ChanceKnapInstance
from src.TimeManager import TimeManager
from src.solver.AdaptivePartitioner import AdaptivePartitioner
from src.solver.MilpSolver import MilpSolver
from src.BigMFinder import BigMFinder

# Expiriment parameters
SEED = 421
NUM_THREADS = 1

# Experiment instance
args = sys.argv[1:]
FILE_LOCATION = args[0]
USE_CONTINUOUS_VAR = (int(args[1]) == 1)
EPSILON = float(args[2])
METHOD = int(args[3])
OUTPUT_FILE_LOCATION = args[4]
TIME_LIMIT = 3600
GAP = 1e-4

# Output selection
WITH_ITERATION_INFO = True

# Setting expiriment parameters
random.seed(SEED)
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)
os.environ['OPENBLAS_NUM_THREADS'] = str(NUM_THREADS)

# Load instance data
chance_instance = ChanceKnapInstance(FILE_LOCATION,
                                     USE_CONTINUOUS_VAR,
                                     EPSILON)

# Creating Output file name and location
file_name = chance_instance.get_file_name()
output_file_name = (OUTPUT_FILE_LOCATION + file_name + "-" +
                    "{:.0f}".format(EPSILON*100) +
                    USE_CONTINUOUS_VAR*"-1-" +
                    (not USE_CONTINUOUS_VAR)*"-0-" +
                    str(METHOD))
computation_output_file_name = output_file_name + ".csv"
iteration_output_file_name = output_file_name + "-iter.csv"

print(computation_output_file_name)


# Timeout handler to stop the process when the time limit is reached
def timeout_handler(signum, frame):
    signal.raise_signal(signal.SIGINT)


# Raise alarm when time limit is reached and call the handler
signal.signal(signal.SIGALRM, timeout_handler)

if METHOD == 1:
    method = MilpSolver(chance_instance, time_limit=TIME_LIMIT, gap=GAP)
    method.solve(use_big_m=True, big_m_method="song",
                 save_bounds=True, path=iteration_output_file_name)
elif METHOD == 2:
    method = MilpSolver(chance_instance, time_limit=TIME_LIMIT, gap=GAP)
    method.solve(use_big_m=True, big_m_method="belotti",
                 save_bounds=True, path=iteration_output_file_name)
elif METHOD == 3:
    signal.alarm(TIME_LIMIT)
    method = AdaptivePartitioner(
        chance_instance, initial_partition_type="random",
        split_method='random',
        projection_method='rescaled_max_violation',
        time_limit=TIME_LIMIT, gap=GAP)
    partitionBigMFinder = BigMFinder(method.chance_instance_part)
    try:
        method.solve(partitionBigMFinder, use_merger=False,
                     use_big_M=True, big_m_method="belotti",
                     use_balancing=False)
    except KeyboardInterrupt:
        print("Reached time limit between iterations.")
elif METHOD == 4:
    signal.alarm(TIME_LIMIT)
    method = AdaptivePartitioner(
        chance_instance, initial_partition_type="cost",
        split_method='cost',
        use_acc_obj=True,
        projection_method='rescaled_max_violation',
        time_limit=TIME_LIMIT, gap=GAP)
    partitionBigMFinder = BigMFinder(method.chance_instance_part)
    try:
        method.solve(partitionBigMFinder, use_merger=True,
                     use_big_M=True, big_m_method="belotti",
                     use_balancing=False, use_lazy=True)
    except KeyboardInterrupt:
        print("Reached time limit between iterations.")

# Writing everything to the file
TimeManager.set_final_time()
method.write_all_computation_details(computation_output_file_name)
if WITH_ITERATION_INFO and METHOD in [3, 4, 5, 6]:
    method.write_iteration_details(iteration_output_file_name)
