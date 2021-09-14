import numpy as np
from scipy import optimize
from projection import projection
from matplotlib import pyplot as plt


#fix seed for debugging purposes
np.random.seed(0)

def mcmc(x0, density,q,dq,d,da,s,N,g,h=None):
    x = x0
    # exp_g = np.array([0]) # the quantity to estimate, e.g. moment of gx
    n0 = 1 # low bound for starting to compute the mean of g
    m = da-d # number of constraints
    xx = np.zeros((da,N))
    gx = np.zeros(N)
    for idx in range(N):
        gx[idx] = g(x)
        # if idx>=n0:
        #     exp_g = np.append(exp_g, exp_g[-1] * (idx - 1) / idx + g(x) / idx)
        dqxT = dq(x).T # basis of Nx


        # generate y
        [Ux,R] = np.linalg.qr(dqxT,'complete') # get orthonormal basis for Tx and Nx
        Nx = Ux[:,:da-d]
        Tx = Ux[:,da-d:]
        vx = Tx @ np.random.normal(0,s,size =(d,1))
        # Sigma=[[0.5,0],[0,0.5]]
        # coo = np.random.multivariate_normal([0,0],Sigma).reshape(-1,1)
        # vx = Tx@coo
        zx = x + vx


        # Newton's method
        nmax = 10
        a0x = np.zeros((m,1)) # a0x lives in orthogonal complement of T_xM
        tol = 1e-3
        ax, flag = projection(zx, a0x, q, dq, dqxT, nmax, tol)

        if flag == 0: y = zx + dqxT @ ax
        else:
            xx[:, idx] = x[:, 0]
            continue

        # check for inequality constraints
        if h is not None:
            if h(y)<=0:
                xx[:, idx] = x[:, 0]
                continue

        #Check for lack of reversibility
        dqyT = dq(y).T
        [Uy,R] = np.linalg.qr(dqyT.reshape((da,m)),'complete')
        Ny = Uy[:,:da-d]
        Ty = Uy[:,da-d:]
        vy = Ty@Ty.T@(x-y) # project x-y onto Ty to find vy
        zy = y + vy

        a0y = np.zeros((m,1))
        b, flag = projection(zy, a0y, q, dq, dqyT, nmax, tol)
        if flag == 1:
            xx[:, idx]= x[:, 0]
            continue


        # MH acceptance-rejectance step
        alpha = min(1,density(y)/density(x)*np.exp(np.linalg.norm(vx)**2-np.linalg.norm(vy)**2))
        u = np.random.uniform(size=1)
        if u<alpha:
            xx[:, idx]= y[:, 0]
            x = y # update the x value
            continue
        else:
            xx[:, idx]= x[:, 0] # this is the only case where we get a new state
    return xx, gx

