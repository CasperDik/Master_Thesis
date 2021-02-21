import numpy as np
import time

tic = time.time()

paths = 5000
mu = 0.06
sigma = 0.5
T = 50
dt = 1/T
S_0 = 10

x = np.exp((mu - sigma ** 2 / 2) * dt + sigma * np.random.normal(0, np.sqrt(dt), size=(paths, T)).T)
x = np.vstack([np.ones(paths), x])
x = S_0 * x.cumprod(axis=0)

toc = time.time()
elapsed_time = toc - tic
print(elapsed_time)

