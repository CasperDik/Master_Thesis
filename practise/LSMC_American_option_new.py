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


def plot_price_matrix(price_matrix, T, ti, paths):
    T = T * ti
    for r in range(paths):
        plt.plot(np.linspace(0, T, T+1), price_matrix[:, r])
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


def plotting_volatility(K, rf, paths, T, ti, mu, sigma, S_0):
    T = T * ti
    tic = time.time()
    for type in ["put", "call"]:
        values = []
        for sig in np.linspace(0, sigma*2, 20):
            price_matrix = GBM(T, ti, paths, mu, sig, S_0)
            value, cf, pv = value_american_option(price_matrix, K, rf, paths, T, ti, type)
            values.append(value)
        plt.plot(np.linspace(0, sigma*2, 20), values, label=type)
    plt.legend()
    plt.title("Option values american call and put options with varying volatility")
    plt.xlabel("Volatility")
    plt.ylabel("Option value")
    plt.show()

    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time for plotting volatility: {:.2f} seconds'.format(elapsed_time))


def plotting_strike(K, rf, paths, T, ti, mu, sigma, S_0):
    T = T * ti
    tic = time.time()
    for type in ["put", "call"]:
        values = []
        for k in np.linspace(K-K/2, K+K/2, 20):
            price_matrix = GBM(T, ti, paths, mu, sigma, S_0)
            value, cf, pv = value_american_option(price_matrix, k, rf, paths, T, ti, type)
            values.append(value)
        plt.plot(np.linspace(K-K/2, K+K/2, 20), values, label=type)
    plt.legend()
    plt.title("Option values american call and put options with varying strike price")
    plt.xlabel("Strike price")
    plt.ylabel("Option value")
    plt.show()

    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time for plotting strike: {:.2f} seconds'.format(elapsed_time))


def value_american_option(price_matrix, K, rf, paths, T, ti, type):
    T = T * ti
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

    for t in range(1, T):
        # find continuation value

        # X = price in time T-1, Y = pv cf
        Y = np.copy(cf_matrix)
        X = np.copy(price_matrix)

        # discount cf 1 period
        Y[T-t] = cf_matrix[T-t+1] * np.exp(-rf*(1/ti))

        # delete columns that are out of the money in T-t
        for j in range(paths-1, -1, -1):
            if price_matrix[T-t, j] * sign > K * sign:
                Y = np.delete(Y, j, axis=1)
                X = np.delete(X, j, axis=1)

        # if at least 1 in the money
        if len(X[T-t]) > 0:
            # regress Y on constant, X, X^2
            regression = np.polyfit(X[T-t], Y[T-t], 2)
            # first is coefficient for X^2, second is coefficient X, third is constant
            beta_2 = regression[0]
            slope = regression[1]
            intercept = regression[2]
            print("Regression: E[Y|X] = ", intercept, " + ", slope, "* X", " + ", beta_2, "* X^2")

        # continuation value
        continuation_value = np.zeros((1, paths))
        tick = 0
        for i in range(paths):
            if price_matrix[T-t, i] * sign <= K * sign:
                continuation_value[0, i] = intercept + slope * X[T-t, i-tick] + beta_2 * (X[T-t, i-tick] ** 2)
            else:
                # add the delete paths back
                continuation_value[0, i] = 0
                tick += 1

        # compare immediate exercise with continuation value
        for i in range(paths):
            if price_matrix[T-t, i] * sign < K * sign:
                # cont > ex --> t=3 is cf exercise, t=2 --> 0
                if continuation_value[0, i] >= payoff_executing(K, price_matrix[T - t, i], type):
                    cf_matrix[T-t, i] = 0
                    # cont < ex --> t=3 is 0, t=2 immediate exercise
                elif continuation_value[0, i] < payoff_executing(K, price_matrix[T - t, i], type):
                    cf_matrix[T-t, i] = payoff_executing(K, price_matrix[T - t, i], type)
                    for l in range(0, t):
                        cf_matrix[T-t+1+l, i] = 0
            # out of the money in t=2, t=2/3 both 0
            else:
                cf_matrix[T-t, i] = 0

    # discounted cash flows
    discounted_cf = np.copy(cf_matrix)
    for t in range(0, T):
        for i in range(paths):
            if discounted_cf[T - t, i] != 0:
                discounted_cf[T - t - 1, i] = discounted_cf[T - t, i] * np.exp(-rf*(1/ti))

    # obtain option value
    option_value = np.sum(discounted_cf[0]) / paths

    print("value of this ", type, " option is: ", option_value)

    # Time and print the elapsed time
    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time: {:.2f} seconds'.format(elapsed_time))

    return option_value, cf_matrix, discounted_cf


# inputs
paths = 2000
# number of years
T = 10
# steps per year
ti = 12

K = 10
S_0 = 12

# yearly
rf = 0.06
q = 0.00
sigma = 0.4
mu = rf-q

price_matrix = GBM(T, ti, paths, mu, sigma, S_0)
# plot_price_matrix(price_matrix, T, ti, paths)
val, cf, pv = value_american_option(price_matrix, K, rf, paths, T, ti, "call")

# plotting_volatility(K, rf, paths, T, ti, mu, sigma, S_0)
# plotting_strike(K, rf, paths, T, ti, mu, sigma, S_0)
