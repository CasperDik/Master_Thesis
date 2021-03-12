import numpy as np
import matplotlib.pyplot as plt
import time

# https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

def MR1(T, dt, paths, sigma, S_0, theta, Sbar):
    Sbar = np.log(Sbar)
    tic = time.time()
    N = T * dt
    N = int(N)
    dt = 1 / dt

    wiener = (sigma * np.random.normal(0, np.sqrt(dt), size=(paths, N+1))).T
    MR_matrix = np.zeros_like(wiener)
    MR_matrix[0] = S_0
    for i in range(1, N+1):
        dx = np.exp(theta * (Sbar - np.log(MR_matrix[i-1])) * dt + wiener[i])
        MR_matrix[i] = MR_matrix[i-1] * dx

    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time of MR1: {:.2f} seconds'.format(elapsed_time))

    return MR_matrix


def MR2(T, dt, paths, sigma, S_0, theta, Sbar):
    tic = time.time()
    N = T * dt
    N = int(N)
    dt = 1 / dt

    wiener = (sigma * np.random.normal(0, np.sqrt(dt), size=(paths, N + 1))).T
    MR_matrix = np.zeros_like(wiener)
    MR_matrix[0] = S_0
    for i in range(1, N + 1):
        dx = np.exp(theta * (np.log(Sbar) - sigma**2/2*theta - np.log(MR_matrix[i - 1])) * dt + wiener[i])
        MR_matrix[i] = MR_matrix[i - 1] * dx

    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time of MR1: {:.2f} seconds'.format(elapsed_time))

    return MR_matrix


def MR3(T, dt, paths, sigma_g, sigma_e, S_0, theta_e, theta_g, Sbar):
    tic = time.time()
    N = T * dt
    N = int(N)
    dt = 1 / dt

    dW_G = (sigma_g * np.random.normal(0, np.sqrt(dt), size=(paths, N + 1))).T
    dW_E = (sigma_e * np.random.normal(0, np.sqrt(dt), size=(paths, N + 1))).T

    # long run equilibrium level
    LR_eq = np.zeros_like(dW_E)
    LR_eq[0] = Sbar #todo: change

    # price matrix
    MR_matrix = np.zeros_like(dW_G)
    MR_matrix[0] = S_0

    for i in range(1, N+1):
        drift = (theta_e * (np.log(Sbar) - sigma**2/2*theta_e - np.log(LR_eq[i-1])))
        LR_eq[i] = LR_eq[i-1] * np.exp(drift * dt + dW_E[i])
        MR_matrix[i] = MR_matrix[i-1] * np.exp((theta_g * (np.log(LR_eq[i]) - sigma_g**2/2*theta_g - np.log(MR_matrix[i-1])))*dt + dW_G[i])

    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time of MR1: {:.2f} seconds'.format(elapsed_time))

    return MR_matrix

T = 1
dt = 365
paths = 14

theta = 0.8
sigma = 0.3
Sbar = 100 # long run equilibrium price
S_0 = 100

# MR1 = MR1(T, dt, paths, sigma, S_0, theta, Sbar)
# MR2 = MR2(T, dt, paths, sigma, S_0, theta, Sbar)
N = T * dt
# plt.plot(np.linspace(0, T, N+1), MR1, label="MR1")
# plt.plot(np.linspace(0, T, N+1), MR2, label="MR1")

# plt.show()


