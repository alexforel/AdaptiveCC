import os
import numpy as np

from src.export.format_results import read_single_experiment, write_table
from src.export.format_results import format_method

# Specify all experimental parameters
ALL_INSTANCES = ["ccmknap-10-10", "ccmknap-20-10", "ccmknap-40-30"]
NB_INSTANCES = 5
ALL_SCENARIOS = [500, 1000, 3000, 5000]
ALL_EPSILONS = [0.1, 0.2]
USE_CONTINUOUS_LIST = [0, 1]
all_instance_ids = range(1, NB_INSTANCES+1)
nb_instances = len(ALL_INSTANCES)
nb_epsilons = len(ALL_EPSILONS)
nb_scenarios = len(ALL_SCENARIOS)
REQUIRED_GAP = 1e-4*100
# Get list of method indices
METHODS = [1, 2, 3, 4]
MILP_METHODS = [1, 2]
PART_METHODS = [3, 4]

# Get path to result files and path to output folder
path_to_root = os.path.dirname(__file__)
input_path_from_root = "/results/tables/"
path_to_read_files = path_to_root + input_path_from_root
path_to_save_files = path_to_root + "/results/"

# Read all individual tables
print('Reading all single-experiment files...')
experiment_dict = dict()
for use_continuous in USE_CONTINUOUS_LIST:
    for i in ALL_INSTANCES:
        for s in ALL_SCENARIOS:
            for j in all_instance_ids:
                for e in ALL_EPSILONS:
                    for m in METHODS:
                        experiment_dict[(use_continuous, i, s, j, e, m)] = (
                            read_single_experiment(path_to_read_files, i, s, j,
                                                   e, use_continuous, m))

# Extract information from individual tables
print('Reading and exporting big-M computation time...')
# Initialize list of lists to store table
summary_tables = [[] for _ in range(nb_instances*nb_scenarios)]
for i, instance in enumerate(ALL_INSTANCES):
    first_time_instance = True
    for s, scenario in enumerate(ALL_SCENARIOS):
        # - First 2 columns: experiment info -
        if first_time_instance:
            print_instance = "\\textsf{"+instance.replace("ccmknap", "mk")+"}"
            first_time_instance = False
        else:
            print_instance = ""
        counter = s + i * nb_scenarios
        summary_tables[counter] = [print_instance, scenario]

        # Method 1: Song Big-M
        # Average over both continuous and binary instances
        big_m_time = 0.0
        m = 0
        method = 1
        # - Iterate over instances to compute table entries -
        for use_continuous in [1, 0]:
            for j in all_instance_ids:
                for _, epsilon in enumerate(ALL_EPSILONS):
                    # Read experiment row from dict
                    experiment_results = experiment_dict[
                        (use_continuous, instance, scenario,
                            j, epsilon, method)]
                    if len(experiment_results) > 0:
                        big_m_time += experiment_results[10]
                    else:
                        big_m_time = np.inf
        big_m_time = big_m_time / (
            len(all_instance_ids) * len(ALL_EPSILONS) * 2)
        # Format for table
        if big_m_time == np.inf:
            big_m_time = "-"
        else:
            big_m_time = "{:.2f}".format(big_m_time)
            big_m_time = big_m_time+'s'
        summary_tables[counter] += [big_m_time]

        # Method 2 and 3: Belotti Big-M
        m = 1
        method = 2
        for use_continuous in [1, 0]:
            big_m_time = 0.0
            # - Iterate over instances to compute table entries -
            for j in all_instance_ids:
                for _, epsilon in enumerate(ALL_EPSILONS):
                    # Read experiment row from dict
                    experiment_results = experiment_dict[
                        (use_continuous, instance, scenario,
                            j, epsilon, method)]
                    if len(experiment_results) > 0:
                        big_m_time += experiment_results[10]
                    else:
                        big_m_time = np.inf
            big_m_time = big_m_time / (
                len(all_instance_ids) * len(ALL_EPSILONS))

            # Format for table
            if big_m_time == np.inf:
                big_m_time = "-"
            else:
                big_m_time = "{:.2f}".format(big_m_time)
                big_m_time = big_m_time+'s'
            summary_tables[counter] += [big_m_time]
print(summary_tables)
offset = nb_scenarios
print("\nFinal tables for big-M times:")
print('------------------------------------')
for i in range(nb_instances):
    instance = ALL_INSTANCES[i]
    file_location2 = (path_to_save_files + instance
                      + "-big-m-times.csv")
    considered_table = summary_tables[offset*i:offset*(1+i)]
    for row in considered_table:
        print(row)
    print('------------------------------------')
    write_table(file_location2, considered_table)

# Extract information from individual tables
print('Processing and formatting results...')
for use_continuous in USE_CONTINUOUS_LIST:
    # Initialize list of lists to store table
    summary_tables = [[] for _ in range(nb_instances*nb_epsilons*nb_scenarios)]
    for i, instance in enumerate(ALL_INSTANCES):
        first_time_instance = True
        for e, epsilon in enumerate(ALL_EPSILONS):
            first_time_epsilon = True
            for s, scenario in enumerate(ALL_SCENARIOS):
                # - First 3 columns: experiment info -
                if first_time_instance:
                    print_instance = ("\\textsf{"
                                      + instance.replace("ccmknap", "mk")
                                      + "}")
                    first_time_instance = False
                else:
                    print_instance = ""
                if first_time_epsilon:
                    print_epsilon = epsilon
                    first_time_epsilon = False
                else:
                    print_epsilon = ""

                counter = s + e * nb_scenarios + i * nb_scenarios * nb_epsilons
                summary_tables[counter] = [print_instance, print_epsilon,
                                           scenario]
                # Initialize arrays to store time and gap of all methods
                m_time = np.ones(len(METHODS)) * np.inf
                m_gap = np.ones(len(METHODS)) * np.inf
                nb_solved = np.ones(len(METHODS)) * np.inf
                for m, method in enumerate(METHODS):
                    # Read experiment row from dict
                    experiment_results = [experiment_dict[
                        (use_continuous, instance, scenario,
                         j, epsilon, method)] for j in all_instance_ids]
                    # Check if this experiment has a file
                    if [] not in experiment_results:
                        m_time[m], nb_solved[m], m_gap[m], new_row = (
                            format_method(experiment_results, method,
                                          all_instance_ids, REQUIRED_GAP,
                                          MILP_METHODS, PART_METHODS))
                    else:
                        m_time[m] = np.inf
                        m_gap[m] = np.inf
                        nb_solved[m] = 0
                        if method in MILP_METHODS:
                            new_row = ["-"]
                        else:
                            new_row = [np.nan, np.nan]
                    summary_tables[counter] += new_row

                # - Format results: highlight best in bold -
                min_m = np.argmin(m_time)
                if not np.isfinite(m_time[min_m]):
                    # Find maximum number of instances solved
                    max_solved = max(nb_solved)
                    valid_idx = np.where(nb_solved == max_solved)[0]
                    min_m = valid_idx[np.argmin(m_gap[valid_idx])]
                if min_m == 0:
                    ind = 3
                elif min_m == 1:
                    ind = 4
                elif min_m == 2:
                    ind = 5
                elif min_m == 3:
                    ind = 7
                elif min_m == 4:
                    ind = 9
                summary_tables[counter][ind] = (
                    "\\textbf{"+str(summary_tables[counter][ind])+"}")

    offset = nb_scenarios*nb_epsilons
    if use_continuous:
        print("\nFinal tables for continuous variables:")
    else:
        print("\nFinal tables for binary variables:")
    for i in range(nb_instances):
        instance = ALL_INSTANCES[i]
        file_location2 = (path_to_save_files + instance + "-" +
                          str(use_continuous) + "-result-table.csv")
        considered_table = summary_tables[offset*i:offset*(1+i)]
        for row in considered_table:
            print(row)
        write_table(file_location2, considered_table)
