from LSMC_American_option_quicker import LSMC, GBM
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
def runtime(K, r, T, type, mu, sigma, S_0):
    n = 20
    s = 40
    Steps = []
    Paths = []
    Time = []
    tick = 0
    for t in np.arange(1, 1 + s*n, s):
        tick += 1
        print(tick, "/", n**2 + n)
        for p in np.arange(100, 100+s*n, s):
            tick += 1
            price_matrix = GBM(T, t, p, mu, sigma, S_0)
            val, time = LSMC(price_matrix, K, r, p, T, t, type)
            Steps.append(t)
            Paths.append(p)
            Time.append(time)
            print(tick, "/", n**2 + n)

    X = []
    X.append(Steps)
    X.append(Paths)
    X = pd.DataFrame(X, dtype=float).transpose()
    Y = pd.DataFrame(Time, dtype=float)

    reg = LinearRegression(fit_intercept=False).fit(X, Y)
    coef = reg.coef_
    print(coef)

    X1 = np.linspace(1, 1 + s*n, s)
    X2 = np.linspace(100, 100+s*n, s)
    Y_hat = coef[0, 0] * X1 + coef[0, 1] * X2
    label = "y = {:.5f}* steps + {:.5f} * paths".format(coef[0,0], coef[0,1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(X[0], X[1], Y, c=Y)
    ax.plot(X1, X2, Y_hat, c="r", label=label)
    plt.colorbar(scatter)
    ax.set_xlabel('Steps')
    ax.set_ylabel('paths')
    ax.set_zlabel('time')
    plt.legend()
    plt.show()

# add elapsed time after return in LSMC function
runtime(10, 0.06, 1, "call", 0.06, 0.2, 10)
