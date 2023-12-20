#include <limits>
#include <cmath>
#include <algorithm>


// Coded and interfaced by Youssouf Emine

struct Comp {
    double* profits;
    double* weights;

    Comp() {
        profits = nullptr;
        weights = nullptr;
    }

    bool operator()(int i, int j) {
        if (profits[i] == 0 && weights[i] == 0) {
            return false;
        } else if (profits[j] == 0 && weights[j] == 0) {
            return true;
        } else {
            return profits[i] * weights[j] > profits[j] * weights[i];
        }
    }
};

void bind(
    Comp& comp,
    double* profits,
    double* weights
) {
    comp.profits = profits;
    comp.weights = weights;
}

void update(
    double* profits, double* loss,
    double* weights, double* capacity,
    int n,
    double* minViolPtr,
    int* indices,
    bool reverse
) {
    double violation = -*loss;
    double weight = 0.0;

    // Add zero-weight items no matter where they are in the indices list
    for (int ii = 0; ii < n; ii+=1) {
        int i = indices[ii];
        if (weights[i] <= 1e-6) {
            violation += profits[i];
        }
    }

    // Add all items in knapsack until capacity is exceeded
    int jstart = reverse ? n - 1 : 0;
    int jend = reverse ? -1 : n;
    int jstep = reverse ? -1 : 1;
    for (int jj = jstart; jj != jend; jj += jstep) {
        int j = indices[jj];
        // Ignore items with zero-weights: they have already been added
        if (weights[j] <= 1e-6) {
            continue;    
        }
        weight += weights[j];
        // Add only residual weight and stop
        if (weight >= (*capacity - 1e-6)) {
            violation += profits[j] * (*capacity - (weight - weights[j])) / weights[j];
            break;
        }
        // Add the entire item to knapsack
        violation += profits[j];
        // Stop adding objects if violation exceeds current min.
        if (violation >= (*minViolPtr + 1e-6)) {
            break;
        }
    }

    if (violation < *minViolPtr) {
        *minViolPtr = violation;
    }
}

void compute(
    double* slhs, double* srhs,
    double* tlhs, double* trhs,
    int n,
    double* sViolPtr,
    double* tViolPtr,
    int* indices,
    Comp& comp
) {
    bind(comp, slhs, tlhs);
    std::sort(indices, indices + n, comp);
    update(slhs, srhs, tlhs, trhs, n, sViolPtr, indices, false);
    update(tlhs, trhs, slhs, srhs, n, tViolPtr, indices, true);
}

void computeAllViolations(
    double* lhs,
    double* rhs,
    int k,
    int n,
    int m,
    double* violations
) {
    double* sViolPtr = nullptr;
    double* tViolPtr = nullptr;
    Comp comp;
    int* indices = new int[n];
    for (int j = 0; j < n; j++) {
        indices[j] = j;
    }

    for (int s = 0; s < k; s++) {
        for (int i = 0; i < m; i++) {
            for (int l = i + 1; l < m; l++) {
                sViolPtr = &violations[s * k * m + s * m + i];
                tViolPtr = &violations[s * k * m + s * m + l];
                compute(
                    &lhs[s * m * n + i * n], &rhs[s * m + i],
                    &lhs[s * m * n + l * n], &rhs[s * m + l],
                    n, sViolPtr, tViolPtr, indices, comp
                );
            }
        }
    }

    for (int s = 0; s < k; s++) {
        for (int t = s + 1; t < k; t++) {
            for (int i = 0; i < m; i++) {
                for (int l = 0; l < m; l++) {
                    sViolPtr = &violations[s * k * m + t * m + i];
                    tViolPtr = &violations[t * k * m + s * m + l];
                    compute(
                        &lhs[s * m * n + i * n], &rhs[s * m + i],
                        &lhs[t * m * n + l * n], &rhs[t * m + l],
                        n, sViolPtr, tViolPtr, indices, comp
                    );
                }
            }
        }
    }
    delete[] indices;
}