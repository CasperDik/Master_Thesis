import numpy as np
import matplotlib.pyplot as plt
import time

# https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

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

    dt = np.array([np.linspace(0,T, N+1)]).T
    ex = np.array(np.exp(-theta*dt))
    ex = np.reshape(ex, (N+1, 1))
    W = np.zeros((N+1, paths))
    for i in range(0, len(W)-1):
        W[i+1] = W[i] + np.sqrt(np.exp(2*theta*dt[i+1])-np.exp(2*theta*dt[i])) * np.random.normal(size=(1, paths))

    MR_matrix = S_0 * ex + Sbar * (1-ex) + sigma * ex * W / np.sqrt(2 * theta)

    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time of MR2: {:.2f} seconds'.format(elapsed_time))

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

# MR2(T, dt, paths, sigma, S_0, theta, Sbar)

"""
links:
https://en.wikipedia.org/wiki/Vasicek_model
https://ipython-books.github.io/134-simulating-a-stochastic-differential-equation/
https://github.com/ipython-books/cookbook-2nd-code/blob/master/chapter13_stochastic/04_sde.ipynb
https://github.com/jwergieluk/ou_noise
https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
"""