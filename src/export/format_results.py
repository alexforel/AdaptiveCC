import numpy as np
import csv


def read_table(file_location):
    with open(file_location, 'r') as file:
        csv_reader = csv.reader(file)
        row = next(csv_reader)
        return row


def write_table(file_location, table):
    with open(file_location, 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(table)


def read_single_experiment(input_path, i, s, j, e, use_continuous, m):
    file_location = (
          input_path + i + "-" +
          str(s) + "-" + str(j) + "-" + str(int(e*100)) + "-" +
          str(use_continuous) + "-" + str(m) + ".csv")
    # Check that file exist, if not print missing file
    try:
        string_table = read_table(file_location)
        real_table = ([string_table[0]]
                      + [float(entry) for entry in string_table[1:]])
    except FileNotFoundError:
        print('File not found:', file_location)
        real_table = []
    return real_table


def format_method(experiment_results, m, all_instance_ids,
                  required_gap, milp_methods, part_methods):
    """
    Read the experiment results for a given method index,
    compute the average time for solving, or average gap
    if not all solved.
    """
    all_solved = True
    nb_solved = 0
    method_time = np.inf
    method_gap = np.inf
    # - Iterate over instances and check if solved -
    for j in all_instance_ids:
        row = experiment_results[j-1]
        considered_gap = row[-1]
        is_solved = (considered_gap <= required_gap + 1e-7)
        if is_solved:
            nb_solved += 1
        else:
            all_solved = False

    # - Creating entry depending on method -
    if m in milp_methods:
        average_table = [0.0]
    else:
        # Adaptive methods: store time and nb of iterations
        average_table = [0.0, 0.0]
    # - Iterate over instances to compute table entries -
    for j in all_instance_ids:
        row = experiment_results[j-1]
        considered_gap = row[-1]
        # Add the correct values to the entries
        if all_solved:
            solving_time = row[-4]
            average_table[0] += solving_time
        else:
            final_gap = row[-1]
            average_table[0] += final_gap
        if m in part_methods:
            average_table[1] += row[-6]

    # - Averaging and adding number of solved -
    average_table = [average_table[entry]/len(all_instance_ids)
                     for entry in range(len(average_table))]
    if all_solved:
        method_time = average_table[0]
    else:
        method_gap = average_table[0]

    # - Format and store results -
    float_entry = "{:.2f}".format(average_table[0])
    if all_solved:
        average_table[0] = float_entry+'s'
    else:
        average_table[0] = float_entry+'\\%('+str(nb_solved)+')'

    return method_time, nb_solved, method_gap, average_table
