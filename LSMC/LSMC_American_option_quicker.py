import numpy as np
import time
import matplotlib.pyplot as plt


def GBM(T, dt, paths, mu, sigma, S_0):
    # start timer
    tic = time.time()
    N = T * dt

    # adjust mu, sigma
    mu = (1+mu)**(1/dt)-1
    sigma = sigma / np.sqrt(dt)

    price_matrix = np.exp((mu - sigma ** 2 / 2) * dt + sigma * np.random.normal(0, np.sqrt(dt), size=(paths, N)).T)
    price_matrix = np.vstack([np.ones(paths), price_matrix])
    price_matrix = S_0 * price_matrix.cumprod(axis=0)

    # Time and print the elapsed time
    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time of GBM: {:.2f} seconds'.format(elapsed_time))

    return price_matrix


def plot_price_matrix(price_matrix, T, dt, paths):
    N = T * dt
    for r in range(paths):
        plt.plot(np.linspace(0, N, N+1), price_matrix[:, r])
        plt.title("GBM")
    plt.show()


def payoff_executing(K, price, type):
    if type == "put":
        payoff_put = K - price
        return payoff_put.clip(min=0)
    elif type == "call":
        payoff_call = price - K
        return payoff_call.clip(min=0)
    else:
        print("Error, only put or call is possible")
        raise SystemExit(0)


def LSMC(price_matrix, K, rf, paths, T, dt, type):
    # start timer
    tic = time.time()

    # total number of steps
    N = T * dt

    # adjust yearly discount factor
    rf = (1+rf)**(1/dt)-1

    # returns -1 if call, 1 for put --> this way the inequality statements can be used for both put and call
    sign = 1
    if type == "call":
        sign = -1

    # cash flow matrix
    cf_matrix = np.zeros((N + 1, paths))

    # calculated cf when executed in time T (cfs European option)
    cf_matrix[N] = payoff_executing(K, price_matrix[N], type)

    # 1 if in the money, otherwise 0
    execute = np.where(payoff_executing(K, price_matrix, type) > 0, 1, 0)

    for t in range(1, N):
        # discounted cf 1 time period
        discounted_cf = cf_matrix[N - t + 1] * np.exp(-rf)

        # slice matrix and make all out of the money paths = 0 by multiplying with matrix "execute"
        X = price_matrix[N - t, :] * execute[N - t, :]

        # +1 here because otherwise will loose an in the money path at T-t,
        # that is out of the money in T-t+1(and thus has payoff=0)
        Y = (discounted_cf+1) * execute[N - t, :]

        # mask all zero values(out of the money paths) and run regression
        X1 = np.ma.masked_less_equal(X, 0)
        Y1 = np.ma.masked_less_equal(Y, 0) - 1

        regression = np.ma.polyfit(X1, Y1, 2)

        # calculate continuation value
        cont_value = np.zeros_like(Y1)
        cont_value = np.polyval(regression, X1)

        # update cash flow matrix
        imm_ex = payoff_executing(K, X1, type)
        cf_matrix[N - t] = np.ma.where(imm_ex > cont_value, imm_ex, 0)
        cf_matrix[N - t + 1:] = np.ma.where(imm_ex > cont_value, 0, cf_matrix[N - t + 1:])

    # todo:  do this without loops?
    discounted_cf = np.copy(cf_matrix)

    for t in range(0, N):
        for i in range(paths):
            if discounted_cf[N - t, i] != 0:
                discounted_cf[N - t - 1, i] = discounted_cf[N - t, i] * np.exp(-rf)

    # obtain option value
    option_value = np.sum(discounted_cf[0]) / paths

    # Time and print the elapsed time
    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time of LSMC: {:.2f} seconds'.format(elapsed_time))

    print("Value of this", type, "option is:", option_value)

    return discounted_cf, cf_matrix


# inputs
"""
# example longstaff and schwartz
price_matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1.09, 1.16, 1.22, 0.93, 1.11, 0.76, 0.92, 0.88], [1.08, 1.26, 1.07, 0.97, 1.56, 0.77, 0.84, 1.22], [1.34, 1.54, 1.03, 0.92, 1.52, 0.9, 1.01, 1.34]])
paths = 8
T = 3
K = 1.1
rf = 0.06
"""

paths = 1000
# years
T = 1
# execute possibilities per year
dt = 365

K = 80
S_0 = 120
rf = 0.06
sigma = 0.1
mu = 0.06

price_matrix = GBM(T, dt, paths, mu, sigma, S_0)
pv, cf = LSMC(price_matrix, K, rf, paths, T, dt, "call")
# plot_price_matrix(price_matrix, T, dt, paths)
