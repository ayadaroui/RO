import numpy as np

def grand_m_method(c, A, b, M=1e6, maximize=True):
    num_constraints, num_vars = A.shape
    A_aug, c_aug, b_aug = np.copy(A), list(c), b.copy()
    artificial_vars, slack_vars = [], []

    for i in range(num_constraints):
        if b[i] < 0: A_aug[i], b_aug[i] = -A_aug[i], -b_aug[i]
        if np.any(A[i] > 0):  # Contrainte "≥"
            A_aug = np.column_stack((A_aug, -np.eye(num_constraints)[:, i]))
            A_aug = np.column_stack((A_aug, np.eye(num_constraints)[:, i]))
            artificial_vars.append(A_aug.shape[1] - 1)
            c_aug += [M, 0]
        else:  # Contrainte "≤"
            A_aug = np.column_stack((A_aug, np.eye(num_constraints)[:, i]))
            c_aug.append(0)

    c_aug = np.array(c_aug)
    tableau = np.zeros((num_constraints + 1, A_aug.shape[1] + 1))
    tableau[:-1, :-1], tableau[:-1, -1], tableau[-1, :-1] = A_aug, b_aug, c_aug
    for var in artificial_vars:
        tableau[-1, :] -= M * tableau[:-1, :][np.where(A_aug[:, var] == 1)[0][0], :]

    while np.min(tableau[-1, :-1]) < 0:
        pivot_col = np.argmin(tableau[-1, :-1])
        ratios = [tableau[i, -1] / tableau[i, pivot_col] if tableau[i, pivot_col] > 0 else np.inf for i in range(num_constraints)]
        pivot_row = np.argmin(ratios)
        tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
        for i in range(num_constraints + 1):
            if i != pivot_row: tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    solution = np.zeros(A_aug.shape[1])
    for i in range(num_constraints):
        idx = np.where(tableau[i, :-1] == 1)[0]
        if len(idx) == 1: solution[idx[0]] = tableau[i, -1]

    optimal_value = -tableau[-1, -1] if maximize else tableau[-1, -1]
    return optimal_value, solution[:num_vars]

# Exemple
c = [-1, -2]  # Maximiser -> minimiser -f(x)
A = np.array([[1, 2],
              [-2, -1]])
b = [8, -6]

# Appel de la méthode du grand M
opt_value, solution = grand_m_method(c, A, b, maximize=False)

# Affichage des résultats
print("Solution optimale :", solution)
print("Valeur optimale :", -opt_value)  # On remet le signe car on a minimisé -f(x)