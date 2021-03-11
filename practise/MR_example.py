import numpy as np
import matplotlib.pyplot as plt
from real_option.MR import MR1


"""
code copied from https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
made some slight adjustment to be able to compare it to my MR code
both models very similar results --> my code also correct (?)
"""

num_sims = 2  # Display five runs

t_init = 0
t_end  = 7
N      = 1000  # Compute 1000 grid points
dt     = float(t_end - t_init) / N
y_init = 0

c_theta = 1.0
c_mu    = 2
c_sigma = 0.3

def mu(y, t):
    """Implement the Ornstein–Uhlenbeck mu."""  # = \theta (\mu-Y_t)
    return c_theta * (c_mu - y)

def sigma(y, t):
    """Implement the Ornstein–Uhlenbeck sigma."""  # = \sigma
    return c_sigma

def dW(delta_t):
    """Sample a random number at each call."""
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))

ts = np.arange(t_init, t_end + dt, dt)
ys = np.zeros(N + 1)

ys[0] = y_init

for _ in range(num_sims):
    for i in range(1, ts.size):
        t = t_init + (i - 1) * dt
        y = ys[i - 1]
        ys[i] = y + mu(y, t) * dt + sigma(y, t) * dW(dt)
    plt.plot(ts, ys, label="MR2")

T=t_end-t_init
MR_matrix = MR1(T, N/T, num_sims, c_sigma, y_init, c_theta, c_mu)
plt.plot(np.linspace(0, T, N+1), MR_matrix, label="MR1")

h = plt.ylabel("y")
h.set_rotation(0)
plt.legend()
plt.show()