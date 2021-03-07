import numpy as np
import matplotlib.pyplot as plt
import time

# Vasicek model
# https://en.wikipedia.org/wiki/Vasicek_model


def MR1(S_0, sigma, dt, T, theta, Sbar):
    tic = time.time()
    N = T * dt
    dt = 1 / dt

    wiener = (sigma * np.random.normal(0, np.sqrt(dt), size=(paths, N))).T
    MR_matrix = np.zeros_like(wiener)
    MR_matrix[0] = S_0
    for i in range(1, N):
        MR_matrix[i] = MR_matrix[i-1] + theta * (Sbar - MR_matrix[i-1]) * dt + wiener[i]

    plt.plot(np.linspace(0, N, N), MR_matrix)
    plt.show()

    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time of LSMC: {:.2f} seconds'.format(elapsed_time))

    return MR_matrix

T = 1
dt = 365
paths = 500

theta = 0.4
sigma = 0.3
Sbar = 100 # long run equilibrium price
S_0 = 100

MR1(S_0, sigma, dt, T, theta, Sbar)
