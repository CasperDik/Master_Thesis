import numpy as np
import matplotlib.pyplot as plt
import time

def generate_random_price_matrix(T, paths, mu, sigma):
    price_matrix = np.zeros(((T+1), paths))
    for t in range(1, T+1):
        for q in range(paths):
            price_matrix[0, q] = 1
            price_matrix[t, q] = max(0, price_matrix[t-1, q] + np.random.normal(mu, sigma, 1))
    return price_matrix

def plot_price_matrix(price_matrix, T, paths):
    for r in range(paths):
        plt.plot(np.linspace(0, T, T+1), price_matrix[:, r])
        plt.title("random generated price series")
    plt.show()

def payoff_executing(K, price, type):
    if type == "put":
        return max(0, K - price)
    elif type == "call":
        return max(0, price - K)
    else:
        print("Error, only put or call is possible")
        raise SystemExit(0)

def plotting_volatility(K, rf, paths, T, mu):
    for type in ["put", "call"]:
        values = []
        for sigma in np.linspace(0, 0.6, 10):
            price_matrix = generate_random_price_matrix(T, paths, mu, sigma)
            value, cf, pv = value_american_option(price_matrix, K, rf, paths, T, type)
            values.append(value)
        plt.plot(np.linspace(0, 0.6, 10), values, label=type)
    plt.legend()
    plt.title("Option values american call and put options with varying volatility")
    plt.xlabel("Volatility")
    plt.ylabel("Option value")
    plt.show()

def plotting_strike(rf, paths, T, mu, sigma):
    for type in ["put", "call"]:
        values = []
        for K in np.linspace(0.8, 1.2, 20):
            price_matrix = generate_random_price_matrix(T, paths, mu, sigma)
            value, cf, pv = value_american_option(price_matrix, K, rf, paths, T, type)
            values.append(value)
        plt.plot(np.linspace(0.8, 1.2, 20), values, label=type)
    plt.legend()
    plt.title("Option values american call and put options with varying strike price")
    plt.xlabel("Strike price")
    plt.ylabel("Option value")
    plt.show()

def value_american_option(price_matrix, K, rf, paths, T, type):
    # start timer
    tic = time.time()

    # cash flow matrix
    cf_matrix = np.zeros((T+1, paths))

    # calculated cf if when executed in time T (cfs European option)
    for p in range(paths):
        cf_matrix[T, p] = payoff_executing(K, price_matrix[T,p], type)
        # cf_matrix[T, p] = max(0, K - price_matrix[T, p])

    for t in range(1, T):
        # find continuation value

        # X = price in time T-1, Y = pv cf
        Y = np.copy(cf_matrix)
        X = np.copy(price_matrix)

        # discount cf t=3 to t=2
        for i in range(paths):
            Y[T-t, i] = cf_matrix[T-t+1, i] * np.exp(-rf)

        # delete columns that are out of the money in T-t
        for j in range(paths-1, -1, -1):
            if price_matrix[T-t, j] > K:
                Y = np.delete(Y, j, axis=1)
                X = np.delete(X, j, axis=1)

        # regress Y on constant, X, X^2
        regression = np.polyfit(X[T-t], Y[T-t], 2)      # todo: when x is zero --> error (print all x? and check when error?)
        # first is coefficient for X^2, second is coefficient X, third is constant
        beta_2 = regression[0]
        slope = regression[1]
        intercept = regression[2]
        print("Regression: E[Y|X] = ", intercept, " + ", slope, "* X", " + ", beta_2, "* X^2")

        # continuation value
        continuation_value = np.zeros((1, paths))
        tick = 0
        for i in range(paths):
            if price_matrix[T-t, i] < K:
                continuation_value[0, i] = intercept + slope * X[T-t, i-tick] + beta_2 * (X[T-t, i-tick] ** 2)
            else:
                # add the delete paths back
                continuation_value[0, i] = 0
                tick += 1

        # compare immediate exercise with continuation value
        for i in range(paths):
            if price_matrix[T-t, i] < K:
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
                discounted_cf[T - t - 1, i] = discounted_cf[T - t, i] * np.exp(-rf)

    # obtain option value
    option_value = np.sum(discounted_cf[0]) / paths

    print("value of this ", type," option is: ", option_value)

    # Time and print the elapsed time
    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time: {:.2f} seconds'.format(elapsed_time))

    return option_value, cf_matrix, discounted_cf

# inputs
paths = 1000
T = 100

K = 1.1
rf = 0.06
mu = 0
sigma = 0.5 / np.sqrt(T)


# price_matrix = generate_random_price_matrix(T, paths, mu, sigma)
# plot_price_matrix(price_matrix, T, paths)
# val, cf, pv = value_american_option(price_matrix, K, rf, paths, T, "call")

plotting_volatility(K, rf, paths, T, mu)
# plotting_strike(rf, paths, T, mu, sigma)