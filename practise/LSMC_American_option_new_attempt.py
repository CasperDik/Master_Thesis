import numpy as np
import matplotlib.pyplot as plt
import time


def GBM(T, ti, paths, mu, sigma, S_0):
    T = T * ti
    dt = 1 / ti

    price_matrix = np.exp((mu - sigma ** 2 / 2) * dt + sigma * np.random.normal(0, np.sqrt(dt), size=(paths, T)).T)
    price_matrix = np.vstack([np.ones(paths), price_matrix])
    price_matrix = S_0 * price_matrix.cumprod(axis=0)

    return price_matrix


def GBM1(T, paths, mu, sigma, S_0):
    price_matrix = np.zeros(((T + 1), paths))
    dt = 1/T
    for q in range(paths):
        price_matrix[0, q] = S_0
        for t in range(1, T+1):
            price_matrix[t, q] = price_matrix[t-1, q] * (1 + (mu * dt + sigma * np.sqrt(dt) * np.random.standard_normal()))
    return price_matrix


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


def american_option(price_matrix, K, rf, paths, T, type):
    # start timer
    tic = time.time()

    # returns -1 if call, 1 for put --> this way the inequality statements can be used for both put and call
    sign = 1
    if type == "call":
        sign = -1

    # cash flow matrix
    cf_matrix = np.zeros((T+1, paths))

    # calculated cf when executed in time T (cfs European option)
    cf_matrix[T] = payoff_executing(K, price_matrix[T], type)

    # 1 if in the money, otherwise 0
    execute = np.where(payoff_executing(K, price_matrix, type) > 0, 1, 0)

    for t in range(1, T):
        # discounted cf 1 time period
        discounted_cf = cf_matrix[T-t+1] * np.exp(-rf)

        # slice matrix and make all out of the money paths = 0 by multiplying with matrix "execute"
        X = price_matrix[T-t, :] * execute[T-t, :]

        # +1 here because otherwise will loose an in the money path at T-t,
        # that is out of the money in T-t+1(and thus has payoff=0)
        Y = (discounted_cf+1) * execute[T-t,:]

        # todo: if len>0
        # mask all zero values(out of the money paths) and run regression
        X1 = np.ma.masked_less_equal(X, 0)
        Y1 = np.ma.masked_less_equal(Y, 0) - 1
        regression = np.ma.polyfit(X1, Y1, 2)

        beta_2 = regression[0]
        slope = regression[1]
        intercept = regression[2]
        print("Regression: E[Y|X] = ", intercept, " + ", slope, "* X", " + ", beta_2, "* X^2")


        # calculate continuation value
        cont_value = np.zeros_like(Y1)
        cont_value = np.polyval(regression, X1)

        # update cash flow matrix
        imm_ex = payoff_executing(K, X1, type)
        cf_matrix[T-t] = np.ma.where(imm_ex * sign >= cont_value * sign, imm_ex, 0)
        cf_matrix[T-t+1] = np.ma.where(imm_ex * sign >= cont_value * sign, 0, cf_matrix[T-t+1])

    # todo:  do this without loops?
    discounted_cf = np.copy(cf_matrix)

    for t in range(0, T):
        for i in range(paths):
            if discounted_cf[T - t, i] != 0:
                discounted_cf[T - t - 1, i] = discounted_cf[T - t, i] * np.exp(-rf)

    # obtain option value
    option_value = np.sum(discounted_cf[0]) / paths
    print("Value of this", type, "option is:", option_value)

    # Time and print the elapsed time
    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time: {:.2f} seconds'.format(elapsed_time))

    return discounted_cf
# inputs

"""
# example longstaff and schwartz
price_matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1.09, 1.16, 1.22, 0.93, 1.11, 0.76, 0.92, 0.88], [1.08, 1.26, 1.07, 0.97, 1.56, 0.77, 0.84, 1.22], [1.34, 1.54, 1.03, 0.92, 1.52, 0.9, 1.01, 1.34]])
paths = 8
T = 3
K = 1.1
rf = 0.06
"""

paths = 2000
T = 10

K = 10
S_0 = 12
rf = 0.06
sigma = 0.4
mu = 0.06

price_matrix = GBM1(T, paths, mu, sigma, S_0)


pv = american_option(price_matrix, K, rf, paths, T, "call")


