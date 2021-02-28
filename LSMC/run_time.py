from LSMC_American_option_quicker import LSMC, GBM
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
def runtime(K, r, T, type, mu, sigma, S_0):
    n = 20
    s = 50
    Steps = []
    Paths = []
    Time = []
    tick = 0
    for t in np.arange(1, 1 + s*n, s):
        tick += 1
        print(tick, "/", n**2 + n)
        for p in np.arange(100, 100+s*n, s):
            tick +=1
            price_matrix = GBM(T, t, p, mu, sigma, S_0)
            time = LSMC(price_matrix, K, r, p, T, t, type)
            Steps.append(t)
            Paths.append(p)
            Time.append(time)
            print(tick, "/",n**2 + n)

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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[0], X[1], Y)
    ax.plot(X1, X2, Y_hat)
    ax.set_xlabel('Steps')
    ax.set_ylabel('paths')
    ax.set_zlabel('time')

    plt.show()


runtime(10, 0.06, 1, "call", 0.06, 0.2, 10)
