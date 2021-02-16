import numpy as np
import matplotlib.pyplot as plt
import time

""" 
Here I have replicated the numerical example in chapter 1 of the paper by Longstaff and Schwartz(2001).
It's an american put option on a non dividend paying share with K=1.1, rf=0.06 
"""

# generate random price matrix
def generate_random_price_matrix(T, paths, mu, sigma):
    price_matrix = np.zeros(((T+1), paths))
    for t in range(1, T+1):
        for q in range(paths):
            price_matrix[0, q] = 1
            price_matrix[t, q] = max(0, price_matrix[t-1, q] + np.random.normal(mu, sigma, 1))
    return price_matrix

# plot of the price series
def plot_price_matrix(price_matrix, T, paths):
    for r in range(paths):
        plt.plot(np.linspace(0, T, T+1), price_matrix[:, r])
        plt.title("random generated price series")
    plt.show()


def value_american_put(price_matrix, K, rf, paths, T):
    # start timer
    tic = time.time()
    # cash flow matrix
    cf_matrix = np.zeros((T+1, paths))

    # calculated cf if when executed in time T (cfs European option)
    for p in range(paths):
        cf_matrix[T, p] = max(0, K - (price_matrix[T, p]))

    for t in range(1, T):
        # find continuation value
        # X = price in time T-1, Y = pv cf
        Y = np.copy(cf_matrix)
        X = np.copy(price_matrix)

        # discount cf t=3 to t=2
        for i in range(paths):
            Y[T-t, i] = cf_matrix[T-t+1, i] * np.exp(-rf)

        # delete columns that are out of the money in t=2
        for j in range(paths-1, -1, -1):
            if price_matrix[T-t, j] > K:
                Y = np.delete(Y, j, axis=1)
                X = np.delete(X, j, axis=1)

        # regress Y on constant, X, X^2
        regression = np.polyfit(X[T-t], Y[T-t], 2)      # todo: what if doesnt converge/no answer
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
                continuation_value[0, i] = 0
                tick += 1

        # compare immediate exercise with continuation value
        for i in range(paths):
            if price_matrix[T-t, i] < K:
                # cont > ex --> t=3 is cf exercise, t=2 --> 0
                if continuation_value[0, i] >= max(0, (K-price_matrix[T-t, i])):
                    cf_matrix[T-t, i] = 0
                    # cont < ex --> t=3 is 0, t=2 immediate exercise
                elif continuation_value[0, i] < max(0, K-price_matrix[T-t, i]):
                    cf_matrix[T-t, i] = max(0, K-price_matrix[T-t, i])
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

    print("value of this put option is: ", option_value)

    # Time and print the elapsed time
    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time: {:.2f} seconds'.format(elapsed_time))

    return cf_matrix, discounted_cf


# stock price matrix as given in Longstaff and Schwartz(2001)
# price_matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1.09, 1.16, 1.22, 0.93, 1.11, 0.76, 0.92, 0.88], [1.08, 1.26, 1.07, 0.97, 1.56, 0.77, 0.84, 1.22], [1.34, 1.54, 1.03, 0.92, 1.52, 0.9, 1.01, 1.34]])

# inputs
paths = 1000
T = 100

K = 0.85
rf = 0.06
mu = 0
sigma = 0.5 / np.sqrt(T)


price_matrix = generate_random_price_matrix(T, paths, mu, sigma)
# plot_price_matrix(price_matrix, T, paths)
cf, pv = value_american_put(price_matrix, K, rf, paths, T)
