from LSMC.LSMC_American_option_quicker import GBM
from real_option.MR import MR2, MR1
import matplotlib.pyplot as plt
import numpy as np

T = 1
dt = 1000
paths = 2
N = T * dt

theta = 1.2
sigma = 0.2
Sbar = 105 # long run equilibrium price
S_0 = 98
mu = 0.03

MR1 = MR1(T, dt, paths, sigma, S_0, theta, Sbar)
MR2 = MR2(T, dt, paths, sigma, S_0, theta, Sbar)
GBM = GBM(T, dt, paths, mu, sigma, S_0)

plt.plot(np.linspace(0, N, N+1), MR1, label="MR1")
plt.plot(np.linspace(0, N, N+1), MR2, label="MR2")
plt.plot(np.linspace(0, N+1, N+1), GBM, label="GBM")

plt.title("MR vs GBM")
plt.legend()
plt.show()
