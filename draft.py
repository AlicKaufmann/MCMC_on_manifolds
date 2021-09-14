import numpy as np
from matplotlib import pyplot as plt

N = 10

z = np.random.uniform(size=N)
mean_est = np.zeros(N+1)
M_est = np.zeros(N+1)
var_est = np.zeros(N+1)
mean_est[0] = z[0]
M_est[0] = 0
var_est[0] = 0

for i in range(len(z)-1):
    mean_est[i+1] = ((i+1)*mean_est[i] + z[i+1])/(i+2)
    M_est[i+1] = M_est[i] + (z[i+1]-mean_est[i])*(z[i+1]-mean_est[i+1])
    var_est[i+1] = M_est[i+1]/(i+2)

pass
