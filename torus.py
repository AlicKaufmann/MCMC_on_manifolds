import numpy as np
from mcmc import mcmc
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


# define constraints
R = 1
r = 0.5
d = 2 # dimension of mf
da = 3 # dimension of ambient space
N = int(1e4)
NN = np.array([100,1000])
s = 0.5 # standart deviation for the movement in the tangent space
Z = 4*np.pi**2*r*R

def q(x):
    x = x.reshape(-1) # transform column vector to 1-D vector
    result = np.array([(R-np.sqrt(x[0]**2+x[1]**2))**2+x[2]**2-r**2])
    result = result[np.newaxis].T # transform 1-D vector to column vector
    return result

def partialx(x,y):
    return (2-2*R/np.sqrt(x**2 + y**2))*x

def dq(x):
    x=x.reshape(-1)
    return np.array([partialx(x[0],x[1]), partialx(x[1],x[0]), 2*x[2]]).reshape(1,-1)

# def dq(x):
#     x =x.reshape(-1)
#     dist = np.sqrt(x[0]**2+x[1]**2)
#     result = np.array([-2*x[0]*(R-dist)/dist, -2*x[1]*(R-dist)/dist, 2*x[2]])
#     if result.ndim == 1:
#         result = result[np.newaxis]
#     return result

x0 = np.array([[1],[0],[0.5]]) # initial state

def f(x): # density function
    return 1

def g(x): # function to compute the moment of inertia
    x = x.flatten()
    return x[0]**2

xx, gx = mcmc(x0=x0,density=f,q=q,dq=dq,d=d,da=da,s=s,N=N,g=g)
inertia = Z*np.mean(gx)

# compute the variance (offline)
mean_mcmc = np.zeros(N+1)
M_mcmc = np.zeros(N+1)
var_mcmc = np.zeros(N+1)
for idx in range(N):
    z = xx[:,idx][0]**2
    # update the mean and the variance
    if idx == 0:
        mean_mcmc[idx] = z

    mean_mcmc[idx+1] = ((idx+1)*mean_mcmc[idx] + z)/(idx+2)
    M_mcmc[idx+1] = M_mcmc[idx] + (z-mean_mcmc[idx])*(z-mean_mcmc[idx+1])
    var_mcmc[idx+1] = M_mcmc[idx+1]/(idx+2)


# ax1 = plt.axes(projection='3d')
# ax1.scatter(xx[0,:],xx[1,:],xx[2,:],s=0.5)
# ax1.set_title("Sampling on torus with the MCMC algorithm")
# plt.show()

# ax2 = plt.axes(projection = None)
# average_update = Z*np.divide(np.cumsum(gx),np.arange(len(gx))+1)
# ax2.plot(average_update, label = "average  inertia moment")
# ax2.set(xlabel = "iteration", ylabel = "average moment")
# print("mcmc estimate for moment: ", average_update[-1])
# ax2.legend()
# plt.show()

# plot the error estimation

def torus_param(theta,phi):
    return np.array([(R+r*np.cos(phi))*np.cos(theta), ((R+r*np.cos(phi))*np.sin(theta)), r*np.sin(phi)])

def phi_ar_sampling():
    f_tilde = lambda phi : 1+r/R*np.cos(phi)
    cg = 1+r/R
    while(True):
        phi = np.random.uniform(0,2*np.pi)
        u = np.random.uniform()
        if u*cg <= f_tilde(phi):
            return phi #accept


moments = np.zeros(N)
phi = np.zeros(N)
theta = 2*np.pi*np.random.uniform(size=N)
param = np.zeros((3,N))
mean_est = np.zeros(N+1)
M_est = np.zeros(N+1)
var_est = np.zeros(N+1)
for idx in range(N):
    phi[idx] = phi_ar_sampling()
    x = torus_param(theta[idx],phi[idx])
    param[:,idx] = x
    z = x[0]**2
    moments[idx] = z

    # update the mean and the variance
    if idx == 0:
        mean_est[idx] = z

    mean_est[idx+1] = ((idx+1)*mean_est[idx] + z)/(idx+2)
    M_est[idx+1] = M_est[idx] + (z-mean_est[idx])*(z-mean_est[idx+1])
    var_est[idx+1] = M_est[idx+1]/(idx+2)


plt.figure()
plt.loglog(np.arange(N+1),var_est*1/np.sqrt(np.arange(1,N+2)), label=r"var/$\sqrt{N}$ Monte Carlo with AR")
plt.loglog(np.arange(N+1),var_mcmc*1/np.sqrt(np.arange(1,N+2)),label=r"var/$\sqrt{N}$ MCMC")
plt.loglog()
plt.loglog(np.arange(1,N+2),np.arange(1,N+2)**-0.5, 'c--', label=r"$1/\sqrt{N}$")
plt.xlabel(r"number of iterations $N$")
plt.ylabel("error")
plt.legend()
inertia_ar = Z*np.mean(moments)
print("acception-rejection estimate for moment: ", inertia_ar)

# ax3 = plt.axes(projection='3d')
# ax3.scatter(param[0,:],param[1,:],param[2,:],s=0.5)
# ax3.set_title("Sampling on torus with accept-reject and crude Monte Carlo")



plt.show()

pass








