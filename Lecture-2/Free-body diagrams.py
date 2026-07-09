"""
Problem Statement : Write a computer program to solve linear algebraic equations to solve a problem associated with the falling parachutist example.
Suppose that a team of three parachutists is connected by a weightless cord while free-falling at a velocity of 5 m/s (Fig.1a). Calculate the tension in each section of cord and the acceleration of the team, given the following:
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
    # Problem. Solve for a, T, R:
    #
    #     70a +  T  +  0R = 636
    #     60a -  T  +  R  = 518
    #     40a +  0  -  R  = 307
    # ------------------------------------------------------------------
    A3 = [[70, 1, 0], [60, -1, 1], [40, 0, -1]]
    b3 = [636, 518, 307]
    a, T, R = gauss_elimination(A3, b3)
    print(f"Problem 3: a={a}, T={T}, R={R}")
