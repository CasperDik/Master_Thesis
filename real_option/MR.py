import numpy as np
import matplotlib.pyplot as plt
import time

# Vasicek model
# https://en.wikipedia.org/wiki/Vasicek_model

def MR1(T, dt, paths, sigma, S_0, theta, Sbar):
    tic = time.time()
    N = T * dt
    N = int(N)
    dt = 1 / dt

    wiener = (sigma * np.random.normal(0, 1, size=(paths, N+1)) * np.sqrt(dt)).T
    MR_matrix = np.zeros_like(wiener)
    MR_matrix[0] = S_0
    for i in range(1, N+1):
        MR_matrix[i] = MR_matrix[i-1] + theta * (Sbar - MR_matrix[i-1]) * dt + wiener[i]

    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time of MR1: {:.2f} seconds'.format(elapsed_time))

    return MR_matrix


def MR2(T, dt, paths, sigma, S_0, theta, Sbar):
    # start timer
    tic = time.time()
    N = T * dt
    N = int(N)
    dt = 1 / dt

    MR_matrix = np.zeros((N+1, paths))
    MR_matrix[0] = S_0
    for t in range(1, N+1):
        MR_matrix[t] = S_0 * np.exp(-theta*t) + Sbar * (1 - np.exp(-theta*t)) + (sigma * np.exp(-theta*t))/(np.sqrt(2 * theta)) * np.random.normal(0, (np.exp(2 * theta * t)-1))

    return MR_matrix

T = 1
dt = 365
paths = 14

theta = 0.8
sigma = 0.3
Sbar = 100 # long run equilibrium price
S_0 = 100

# MR_matrix = MR1(T, dt, paths, sigma, S_0, theta, Sbar)
N = T * dt
# plt.plot(np.linspace(0, T, N+1), MR_matrix, label="MR1")
# plt.show()

"""
links:
https://en.wikipedia.org/wiki/Vasicek_model
https://ipython-books.github.io/134-simulating-a-stochastic-differential-equation/
https://github.com/ipython-books/cookbook-2nd-code/blob/master/chapter13_stochastic/04_sde.ipynb
https://github.com/jwergieluk/ou_noise
https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
"""