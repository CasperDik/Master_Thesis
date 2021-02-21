import numpy as np
import time

tic = time.time()

mu = 1
sigma = 0.5
T = 5000
dt = 1/T
S_0 = 10

paths = np.arange(1, 5000, 1)

x = np.exp((mu - sigma ** 2 / 2) * dt + sigma * np.random.normal(0, np.sqrt(dt), size=(len(paths), T)).T)
x = np.vstack([np.ones(len(paths)), x])
x = S_0 * x.cumprod(axis=0)

toc = time.time()
elapsed_time = toc - tic
print(elapsed_time)

