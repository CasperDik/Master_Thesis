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

    wiener = (sigma * np.random.normal(0, np.sqrt(dt), size=(paths, N+1))).T
    MR_matrix = np.zeros_like(wiener)
    MR_matrix[0] = S_0
    for i in range(1, N+1):
        # MR_matrix[i] = MR_matrix[i-1] + theta * (Sbar - MR_matrix[i-1]) * dt + wiener[i]
        drift = theta * (Sbar - MR_matrix[i-1])
        vol = wiener[i] * MR_matrix[i-1]
        MR_matrix[i] = drift + vol + MR_matrix[i-1]

    # plt.plot(np.linspace(0, N+1, N+1), MR_matrix)
    # plt.show()

    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time of MR1: {:.2f} seconds'.format(elapsed_time))

    return MR_matrix

def MR2(T, dt, paths, sigma, S_0, theta, Sbar):
    """modeled as in pindyck 1994, investment under uncertainty, around p.76"""
    tic = time.time()
    N = T * dt
    N = int(N)
    dt = 1 / dt

    variance = (sigma**2/2)*theta * (1 - np.exp(-2*theta))
    epsilon =  (np.random.normal(0, np.sqrt(variance), size=(paths, N+1))).T
    MR_matrix = np.zeros_like(epsilon)
    MR_matrix[0] = S_0

    for i in range(1, N + 1):
        change = Sbar*(1-np.exp(-theta))+(np.exp(-theta)-1)*MR_matrix[i-1] + epsilon[i]
        MR_matrix[i] = MR_matrix[i-1] + change

    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time of MR1: {:.2f} seconds'.format(elapsed_time))

    return MR_matrix


T = 1
dt = 365
paths = 14

theta = 0.4
sigma = 0.3
Sbar = 100 # long run equilibrium price
S_0 = 100

# MR1(T, dt, paths, sigma, S_0, theta, Sbar)


"""
links:
https://en.wikipedia.org/wiki/Vasicek_model
https://ipython-books.github.io/134-simulating-a-stochastic-differential-equation/
https://github.com/ipython-books/cookbook-2nd-code/blob/master/chapter13_stochastic/04_sde.ipynb
https://github.com/jwergieluk/ou_noise
https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
"""