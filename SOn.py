import numpy as np
from mcmc import mcmc
from matplotlib import pyplot as plt
import scipy.stats as stats

n = 11
x0 = np.eye(n,n).reshape(-1,1)
da = n**2
d = da - int(n*(n+1)/2)
N = 4000
# s = 0.28 # standard deviation when searching in the tangent space
s = 0.1

def delta(i,j):
    if i != j:
        return 0
    else:
        return 1

def q(x):
    constr = np.zeros((n,n))
    X = x.reshape(n,n)
    for k in range(n):
        constr[k, k] = sum(X[k, :]**2)-1
        for l in range(k+1,n):
            constr[k,l] = sum(X[k,:]*X[l,:])

    r = np.array([])
    for idx in range(n):
        r = np.hstack((r,constr[idx,idx:]))

    r = r.reshape(-1,1)

    return r

def dq(x):
    deriv =  np.zeros((n,n,n,n)) # 4-th order tensor to store the derivative
    X = x.reshape(n,n)
    for k in range(n):
        for l in range(n):
            for i in range(n):
                for j in range(n):
                    deriv[k,l,i,j] = X[l,j]*delta(k,i)+X[k,j]*delta(l,i)

    r = np.array([])
    for k in range(n):
        for l in range(k,n):
            r = np.append(r,deriv[k,l].flatten())

    r = r.reshape(int(n*(n+1)/2),n**2)
    return r

# do mcmc on manifold

def tr(x):
    x = x.reshape(n,n)
    return np.trace(x)

def determinant(x):
    x = x.reshape(n,n)
    return np.linalg.det(x)

def f(x): #density function
    return 1

xx, trx = mcmc(x0=x0, density=f,q=q,dq=dq,d=d,da=da,s=s,N=N,g=tr,h=determinant)
trx_last = trx[int(N/10):]

plt.hist(trx_last, bins = 25, density = True, label="trace sample distr.")
x = np.linspace(-4,4,100)
plt.plot(x, stats.norm.pdf(x, 0, 1), label = "trace theoretical pdf")
plt.xlabel("trace")
plt.ylabel("density")
plt.legend()
plt.show()

# A = np.array([[np.sqrt(3)/2,-1/2],[1/2,np.sqrt(3)/2]])
# print(q(A,2))
# print(dq(A,2))



pass
