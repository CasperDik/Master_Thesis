from real_option.MR import MR2, MR3
from real_option.RO_LSMC import GBM, LSMC_RO
from LSMC.LSMC_American_option_faster import LSMC

# todo: make this RO setting
# todo: make this function, so can import it to different file for plotting

# inputs
T = 2
dt = 365
paths = 1000
N = T * dt

theta = 1.1
sigma = 0.2
Sbar = 100 # long run equilibrium price
S_0 = 100
mu = 0.2

sigma_g = 0.2
sigma_e = 0.2
theta_e = 1.1
theta_g = 1.1


# stochastic processes
GBM = GBM(T, dt, paths, mu, sigma, S_0)
MR2 = MR2(T, dt, paths, sigma, S_0, theta, Sbar)
MR3 = MR3(T, dt, paths, sigma_g, sigma_e, S_0, theta_e, theta_g, Sbar)

# option values
K = 100
r = 0.03

print("value option with GBM:", LSMC(GBM, K, r, paths, T, dt, "call"))
print("value option with simple MR:", LSMC(MR2, K, r, paths, T, dt, "call"))
print("value option with bivariate MR:", LSMC(MR3, K, r, paths, T, dt, "call"))
