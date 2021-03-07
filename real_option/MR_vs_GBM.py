from LSMC.LSMC_American_option_quicker import GBM, LSMC
from real_option.MR1 import MR1
import matplotlib.pyplot as plt
import numpy as np

T = 1
dt = 365
paths = 2
N = T * dt

theta = 0.00
sigma = 0.3
Sbar = 100 # long run equilibrium price
S_0 = 100

mu = 0.00
r = 0.05
K = 100

MR = MR1(T, dt, paths, sigma, S_0, theta, Sbar)
GBM = GBM(T, dt, paths, mu, sigma, S_0)
LSMC(MR, K, r, paths, T, dt, "call")
LSMC(GBM, K, r, paths, T, dt, "call")

plt.plot(np.linspace(0, N+1, N+1), MR, label="MR")
plt.plot(np.linspace(0, N+1, N+1), GBM, label="GBM")

plt.legend()
plt.show()
