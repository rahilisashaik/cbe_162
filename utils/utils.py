import numpy as np
import sympy as sp

def construct_routh_array(coefficients):
    """
    Constructs the Routh array for a given characteristic equation.
    :param coefficients: List of coefficients of the characteristic equation (highest degree first)
    :return: The Routh array as a 2D numpy array
    """
    n = len(coefficients)
    row_count = n if n % 2 == 0 else n + 1
    routh_array = np.zeros((row_count, (n + 1) // 2))

    routh_array[0, :len(coefficients[0::2])] = coefficients[0::2]
    routh_array[1, :len(coefficients[1::2])] = coefficients[1::2]

    for i in range(2, n):
        for j in range(0, routh_array.shape[1] - 1):
            numerator = (routh_array[i - 1, 0] * routh_array[i - 2, j + 1] -
                         routh_array[i - 2, 0] * routh_array[i - 1, j + 1])
            denominator = routh_array[i - 1, 0]
            routh_array[i, j] = numerator / denominator if denominator != 0 else 0

        if np.allclose(routh_array[i, :], 0):
            routh_array[i, :] = np.zeros_like(routh_array[i, :])
            break

    return routh_array[:n]  # Return only non-zero rows

def verify_stability(routh_array):
    """
    Verifies stability using the Routh Stability Criterion.
    :param routh_array: The Routh array as a 2D numpy array
    :return: Boolean indicating stability
    """
    first_column = routh_array[:, 0]
    return np.all(first_column > 0)

def solve_system(equations, variables):
    """
    Solves a system of equations for the given variables.

    Parameters:
    - equations: A list of equations (e.g., [eq1, eq2, eq3])
    - variables: A list of unknowns to solve for (e.g., [x, y, z])

    Returns:
    - A dictionary mapping each variable to its solution.
    """
    solutions = sp.solve(equations, variables, dict=True)
    return solutions


def main():
    # example usage
    characteristic_eq = [1, 6, 11, 6]
    routh_array = construct_routh_array(characteristic_eq)
    print("Routh Array:")
    print(routh_array)

    is_stable = verify_stability(routh_array)
    if is_stable:
        print("The system is stable.")
    else:
        print("The system is unstable.")

if "__name__" == "__main__":
    main()