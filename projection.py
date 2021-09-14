import numpy as np

def projection(z, a0, q, dq, dqxT, nmax,tol):
    flag = 0
    a = a0
    counter = 0

    while np.linalg.norm(q(z + dqxT @ a)) >= tol:
        jac = np.einsum('ik,kj->ij', dq(z + dqxT @ a), dqxT) #something weird with the sign
        a = a - np.linalg.solve(jac, q(z + dqxT @ a))
        if counter>=nmax:
            flag = 1
            return a, flag
        counter = counter + 1
    return a, flag
