"""Utility functions for linear algebra"""
import numpy as np
import scipy
from scipy.linalg import svd

def count_nonzero(A, axis=0):
    """
    Counts the nonzero elements along the specified `axis` (default is to count along rows).
    Args:
        A (np.ndarray): a matrix
        axis (int)
    Returns:
        (np.ndarray) a flattened 1-d array of counts
    """
    if not (axis == 0 or axis == 1):
        raise ValueError

    tol = 1e-13
    return np.array((abs(A) > tol).sum(axis=1-axis)).flatten()


def get_densest(A, elibible_rows):
    """
    Returns the index of the densest row of `A`. Ignores rows that are not eligible for consideration.
    Args:
        A (np.ndarray): a matrix
        elibible_rows (np.ndarray): 1-D array of boolean values which indicate whether the corresponding row of `A`
            is eligible for consideration
    Returns:
        idx_densest (int): the index of the densest row in `A` among eligible candidates
    """
    row_counts = count_nonzero(A)
    return np.argmax(row_counts * elibible_rows)


def remove_zero_rows(A, b):
    """
    Removes zero rows from a system of equations Ax = b. Also, checks for trivial infeasibility.
    (adapted from https://github.com/JayMan91/NeurIPSIntopt/blob/main/Interior/remove_redundancy.py#L57)
    Args:
        A (np.ndarray): a matrix of coefficients
        b (np.ndarray): RHS
    Returns:
        A (np.ndarray): an array with zero rows removed
        b (np.ndarray): the RHS with corresponding rows removed
        status (int): an integer indicating the status of teh removal (0: feasible, 2: trivially infeasible)
        message (str): a string descriptor of the exit status of the operation.
    """
    status = 0
    message = ""
    idx_zero = count_nonzero(A, axis=0) == 0  # Indices of zero rows
    A = A[np.logical_not(idx_zero), :]
    if not (np.allclose(b[idx_zero], 0)):
        status = 2
        message += "There is a zero row in `A` whose associated `b` is nonzero. The problem is infeasbile."
    b = b[np.logical_not(idx_zero)]
    return A, b, status, message


def remove_redundancy_dense(A, rhs):
    """
    Eliminates redundant equations from the system of equations `Ax = b` and identifies infeasibility.
    (adapted from https://github.com/JayMan91/NeurIPSIntopt/blob/main/Interior/remove_redundancy.py#L107)
    Args:
        A (np.ndarray): 2-D dense matrix
        rhs (np.ndarray): 1-D array
    Returns:
        A (np.ndarray): 2-D dense matrix representing the coefficient matrix of the system of equations
        rhs (np.ndarray): 1-D array representing the RHS of the system of equations
        status (int): an integer indicating the property of the system (0: feasible, 2: trivially infeasible)
        message (str): a string descriptor of the exit status of the operation.
    """
    tolapiv = 1e-8
    tolprimal = 1e-8
    status = 0
    message = ""
    inconsistent = ("There is a linear combination of rows of `A` that "
                    "results in zero, suggesting a redundant constraint. "
                    "However, the same linear combination of `b` is "
                    "nonzero, suggesting that the constraints conflict "
                    "and the problem is infeasible.")
    A, rhs, status, message = remove_zero_rows(A, rhs)

    if status != 0:
        return A, rhs, status, message

    m, n = A.shape

    v = list(range(m))      # Initial `m` column indices
    b = v.copy()            # Basis column indices
    k = set(range(m, m+n))  # Structural column indices
    d = []
    lu = None
    perm_r = None

    A_orig = A
    A = np.hstack((np.eye(m), A))
    e = np.zeros(m)

    def bg_update_dense(plu, perm_r, v, j):
        LU, p = plu

        u = scipy.linalg.solve_triangular(LU, v[perm_r], lower=True, unit_diagonal=True)
        LU[:j + 1, j] = u[:j + 1]
        l = u[j + 1:]
        piv = LU[j, j]
        LU[j + 1:, j] += (l / piv)
        return LU, p

    B = A[:, b]
    for i in v:

        e[i] = 1
        if i > 0:
            e[i - 1] = 0

        try:  # fails for i==0 and any time it gets ill-conditioned
            j = b[i - 1]
            lu = bg_update_dense(lu, perm_r, A[:, j], i - 1)
        except Exception:
            lu = scipy.linalg.lu_factor(B)
            LU, p = lu
            perm_r = list(range(m))
            for i1, i2 in enumerate(p):
                perm_r[i1], perm_r[i2] = perm_r[i2], perm_r[i1]

        pi = scipy.linalg.lu_solve(lu, e, trans=1)

        # not efficient, but this is not the time sink...
        js = np.array(list(k - set(b)))
        batch = 50

        # This is a tiny bit faster than looping over columns indivually,
        # like for j in js: if abs(A[:,j].transpose().dot(pi)) > tolapiv:
        for j_index in range(0, len(js), batch):
            j_indices = js[np.arange(j_index, min(j_index + batch, len(js)))]

            c = abs(A[:, j_indices].transpose().dot(pi))
            if (c > tolapiv).any():
                j = js[j_index + np.argmax(c)]  # very independent column
                B[:, i] = A[:, j]
                b[i] = j
                break
        else:
            bibar = pi.T.dot(rhs.reshape(-1, 1))
            bnorm = np.linalg.norm(rhs)
            if abs(bibar) / (1 + bnorm) > tolprimal:  # inconsistent
                status = 2
                message = inconsistent
                return A_orig, rhs, status, message
            else:  # dependent
                d.append(i)

    keep = set(range(m))
    keep = list(keep - set(d))
    return A_orig[keep, :], rhs[keep], status, message


def remove_redundancy(A, rhs):
    """
    Eliminates redundant equations from the system of equations `Ax = b` and identifies infeasibility.
    (adapted from https://github.com/JayMan91/NeurIPSIntopt/blob/main/Interior/remove_redundancy.py#L360)
    Args:
        A (np.ndarray): 2-D dense matrix
        rhs (np.ndarray): 1-D array
    Returns:
        A (np.ndarray): 2-D dense matrix representing the coefficient matrix of the system of equations
        rhs (np.ndarray): 1-D array representing the RHS of the system of equations
        status (int): an integer indicating the property of the system (0: feasible, 2: trivially infeasible)
        message (str): a string descriptor of the exit status of the operation.
    """
    status = 0
    message = ""

    A_ = A.copy()
    m, n = A_.shape
    eps = np.finfo(float).eps

    U, s, Vh = svd(A_)
    s_min = s[-1] if m <= n else 0
    tol = s.max() * max(A.shape) * eps
    d = []

    # this algorithm is faster than that of [2] when the nullspace is small
    # but it could probably be improvement by randomized algorithms and with
    # a sparse implementation.
    # it relies on repeated singular value decomposition to find linearly
    # dependent rows (as identified by columns of U that correspond with zero
    # singular values). Unfortunately, only one row can be removed per
    # decomposition (I tried otherwise; doing so can cause problems.)
    # It would be nice if we could do truncated SVD like sp.sparse.linalg.svds
    # but that function is unreliable at finding singular values near zero.
    # Finding max eigenvalue L of A A^T, then largest eigenvalue (and
    # associated eigenvector) of -A A^T + L I (I is identity) via power
    # iteration would also work in theory, but is only efficient if the
    # smallest nonzero eigenvalue of A A^T is close to the largest nonzero
    # eigenvalue.

    while abs(s_min) < tol:
        v = U[:, -1]  # TODO: return these so user can eliminate from problem?
        # rows need to be represented in significant amount
        eligibleRows = np.abs(v) > tol * 10e6
        # if not np.any(eligibleRows) or np.any(np.abs(v.dot(A)) > tol):
        #     status = 4
        #     message = ("Due to numerical issues, redundant equality "
        #                "constraints could not be removed automatically. "
        #                "Try providing your constraint matrices as sparse "
        #                "matrices to activate sparse presolve, try turning "
        #                "off redundancy removal, or try turning off presolve "
        #                "altogether.")
        #     break
        # if np.any(np.abs(v.dot(b)) > tol * 100):  # factor of 100 to fix 10038 and 10349
        #     status = 2
        #     message = ("There is a linear combination of rows of A_eq that "
        #                "results in zero, suggesting a redundant constraint. "
        #                "However the same linear combination of b_eq is "
        #                "nonzero, suggesting that the constraints conflict "
        #                "and the problem is infeasible.")
        #     break

        i_remove = get_densest(A_, eligibleRows)
        A_ = np.delete(A_, i_remove, axis=0)
        # b = np.delete(b, i_remove)
        U, s, Vh = svd(A_)
        m, n = A_.shape
        s_min = s[-1] if m <= n else 0
        d.append(i_remove)

    return d, status, message