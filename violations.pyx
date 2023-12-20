import ctypes
import numpy as np

cdef extern from "violations.h":
    void computeAllViolations(double*, double*, int, int, int, double*)

cpdef compute_all_violations(double [:,:,::1] A, double [:,::1] b):
    cdef int k = int(A.shape[0])
    cdef int m = int(A.shape[1])
    cdef int n = int(A.shape[2])
    cdef double [:,:,::1] viols = np.ascontiguousarray(np.ones((k, k, m)) * np.inf)

    # Initialize diagonal elements with
    for s in range(k):
        for i in range(m):
            viols[s, s, i] = 0.0

    # Call c++ function
    computeAllViolations(&A[0, 0, 0], &b[0, 0], k, n, m, &viols[0, 0, 0])

    return np.asarray(viols)