import numpy as np


def solve_continuous_knapsack(revenue, obj_constant, weights, capacity):
    """ Add items until the knapsack constraint is tight. """
    profit = -obj_constant

    # Add all items with zero weight
    zero_indices = np.where(weights == 0.0)[0]
    for i in zero_indices:
        profit += revenue[i]

    revenue = revenue[weights != 0.0]
    weights = weights[weights != 0.0]
    # Compute the items' marginal values
    values = np.divide(revenue, weights)
    ordered_values_index = np.argsort(-values)

    knapsack_weight = 0.0
    for i in ordered_values_index:
        available_capacity = capacity - knapsack_weight
        if available_capacity > 0.0:
            quantity = min(available_capacity / weights[i], 1.0)
            assert quantity >= 0.0
            if revenue[i] > 0.0:
                profit += revenue[i] * quantity
                knapsack_weight += weights[i] * quantity
        else:
            break

    return profit


def song_min_s_prime_violation(nb_constraints, A_sprime, b_sprime, A_si, b_si):
    violation = np.inf
    for iprime in range(nb_constraints):
        lhs_vector = A_sprime[iprime, :]
        rhs = b_sprime[iprime]
        # Solve continuous single-dimensional knapsack
        temp_violation = solve_continuous_knapsack(A_si, b_si, lhs_vector, rhs)
        violation = min(violation, temp_violation)
    return violation


def single_scenario_iter(A_si, b_si, A_matrices, b_vectors,
                         nb_constraints, nb_scenarios):
    violations = np.zeros(nb_scenarios)
    for sprime in range(nb_scenarios):
        A_sprime = A_matrices[sprime, :, :]
        b_sprime = b_vectors[sprime, :]
        violation = song_min_s_prime_violation(nb_constraints, A_sprime,
                                               b_sprime, A_si, b_si)
        violations[sprime] = violation
    return violations


def compute_all_violations(A_matrices, b_vectors):
    """Song et al method to compute big Ms.

    This is an inefficient Python implementation.
    It is only used in the unit tests.
    """
    nb_scenarios = A_matrices.shape[0]
    nb_constraints = A_matrices.shape[1]
    # Initialize the numpy array to store the (s, i, sprime) violations
    song_violations = np.ones(
        (nb_scenarios, nb_scenarios, nb_constraints)) * np.inf

    for s in range(nb_scenarios):
        print('Song bigM -> iterating over all scenarios: [%d%%] \r'
              % (100*(s+1)//nb_scenarios), end="")
        for i in range(nb_constraints):
            A_si = A_matrices[s, i, :]
            b_si = b_vectors[s, i]
            song_violations[s, :, i] = single_scenario_iter(
                A_si, b_si, A_matrices, b_vectors,
                nb_constraints, nb_scenarios)
    return song_violations
