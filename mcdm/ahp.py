import numpy as np

# Function to calculate priority vector (vp) and consistency ratio (C)
def ahp_vp(P):
    # If P is a vector, convert it to a square diagonal matrix
    if P.ndim == 1:
        raise ValueError("Error! Input matrix P must be at least two-dimensional.")

    # Calculate eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(P)

    # Find the index of the largest eigenvalue
    max_index = np.argmax(eigvals)

    # Take the eigenvector associated with the largest eigenvalue
    vl = eigvecs[:, max_index]

    # Normalize the priority vector
    vp = vl / np.sum(vl)

    # Maximum eigenvalue
    lmax = eigvals[max_index]
    n= P.shape[0]
    # Consistency ratio
    C = (lmax -n) / (n - 1)

    return vp.real, C.real

# Main AHP function
def ahp(P, *args):
    vpa, Ca = ahp_vp(P)

    ncrit = P.shape[0]  # Number of criteria
    nalt = 0  # Number of alternatives
    mpa = np.array([])  # Priority matrix for alternatives
    Caa = []  # List of consistency ratios for alternatives

    # Loop through all input data
    for i, arg in enumerate(args):
        pp = np.array(arg)

        # If input is quantitative (one-dimensional vector)
        if pp.ndim == 1:
            sum_val = np.sum(pp)
            vpi = pp / sum_val  # Normalize the quantitative vector
            Cai = 0  # Consistency ratio for quantitative data
        else:
            # If the input is a matrix, apply AHP methodology
            vpi, Cai = ahp_vp(pp)

        # Initialize priority matrix in the first iteration
        if i == 0:
            mpa = vpi.reshape(-1, 1)
            Caa.append(Cai)
        else:
            mpa = np.column_stack((mpa, vpi))
            Caa.append(Cai)

        # Set number of alternatives based on the first matrix
        if i == 0:
            nalt = pp.shape[0]

    # Final priority vector
    vp = np.dot(mpa, vpa)

    # Saaty's random index for consistency ratio (RI)
    ri = [0.58, 0.9, 1.12,1.24,1.32,1.41,1.45,1.49]

    # Calculate CR (Consistency Ratio) for criteria and alternatives
    cr = (Ca / ri[ncrit-3])
    cri = [(caa / ri[nalt-3])  for caa in Caa]

    # Global consistency: CR for criteria plus weighted sum of CR for alternatives
    global_cr = cr + sum(vpa[i] * cri[i] for i in range(len(cri)))


    cr = round(float(cr), 4)
    cri = [round(float(c), 4) for c in cri]
    global_cr = round(float(global_cr), 4)

    return vp, vpa, mpa, Ca, Caa, cr, cri, global_cr


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example with input data
    attribs = ['Design', 'Reliability', 'Consumption']
    variants = ['Pontiac', 'Volvo', 'Mercedes', 'Ford']

    # Criteria
    P = np.array([[1, 1 / 2, 3],
                  [2, 1, 4],
                  [1/3, 1/4, 1]])

    # Design
    P_a1 = np.array([[1, 1 / 4, 4, 1 / 6],
                     [4, 1, 4, 1/4],
                     [1/4, 1/4, 1, 1/5],
                     [6, 4, 5, 1]])

    # Reliability
    P_a2 = np.array([[1, 2, 5, 1],
                     [1/2, 1, 3, 2],
                     [1/5, 1/3, 1, 1/4],
                     [1, 1/2, 4, 1]])

    # Fuel Efficiency (quantitative data)
    Q_a3 = np.array([34, 27, 24, 28])

    # Call the AHP function
    vp, vpa, mpa, Ca, Caa, cr, cri, global_cr = ahp(P, P_a1, P_a2, Q_a3)

    # Print results
    print("Priority vector:", vp)
    print("Consistency ratio for criteria:", cr)
    print("Consistency ratios for alternatives:", cri)
    print("Global consistency ratio:", global_cr)

    # Plot pie charts for criteria and variants
    plt.figure(1)
    plt.pie(vpa, labels=attribs, autopct='%1.1f%%')
    plt.title('Criteria')

    plt.figure(2)
    plt.pie(vp, labels=variants, autopct='%1.1f%%')
    plt.title('Alternatives')

    plt.show()


