"""
Gauss elimination with partial pivoting.

Solves Ax = b by reducing the augmented matrix [A|b] to upper-triangular form (with partial pivoting to guard against small/zero pivots), then finishing with back substitution.
"""

def gauss_elimination(A, b):
    n = len(A)
    # augmented matrix: a[i] = row i of A, plus b[i] appended as column n
    a = [row[:] + [b[i]] for i, row in enumerate(A)]

    for k in range(n - 1):                    # k = 0 .. n-2 (steps 1-4)
        # Step 1-2: partial pivoting -- bring the largest |a[j][k]| to row k
        pivot_row = max(range(k, n), key=lambda j: abs(a[j][k]))
        if abs(a[pivot_row][k]) == 0:
            raise ValueError("No unique solution exists.")
        a[k], a[pivot_row] = a[pivot_row], a[k]

        # Steps 3-4: eliminate a[j][k] to zero for every row below the pivot
        for j in range(k + 1, n):
            m = a[j][k] / a[k][k]
            for p in range(k, n + 1):
                a[j][p] -= m * a[k][p]

    # Step 5: final singularity check
    if a[n - 1][n - 1] == 0:
        raise ValueError("No unique solution exists.")

    # Steps 6-7: back substitution, from x[n-1] up to x[0]
    x = [0.0] * n
    x[n - 1] = a[n - 1][n] / a[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        s = a[i][n] - sum(a[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / a[i][i]

    return x


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Problem 1 Solve:
    #
    #     3x1 + 5x2 +  x3 = 16
    #      x1 + 4x2 + 2x3 = 15
    #     2x1 + 2x2 + 3x3 = 15
    #
    # Expected solution: x = [1, 2, 3]
    # ------------------------------------------------------------------
    A1 = [[3, 5, 1], [1, 4, 2], [2, 2, 3]]
    b1 = [16, 15, 15]
    print("Problem 1:", gauss_elimination(A1, b1))

    # ------------------------------------------------------------------
    # Problem 2 (requires pivoting). Solve:
    #
    #      x1 - 2x2 +  x3 + 2x4 =  8
    #     2x1 + 3x2 - 2x3 + 3x4 = 14
    #     4x1 -  x2 + 3x3 -  x4 =  7
    #     3x1 + 2x2 - 4x3 + 5x4 =  5
    #
    # Expected solution: x ~= [-1.1053, 4.632, 7.2105, 5.5789]
    # ------------------------------------------------------------------
    A2 = [[1, -2, 1, 2], [2, 3, -2, 3], [4, -1, 3, -1], [3, 2, -4, 5]]
    b2 = [8, 14, 7, 5]
    print("Problem 2:", gauss_elimination(A2, b2))

