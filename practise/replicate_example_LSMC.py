import numpy as np

""" 
Here I have replicated the numerical example in chapter 1 of the paper by Longstaff and Schwartz(2001) 
"""

# stock price paths matrix
# 8 paths, from t=0 to t=3
price_matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1.09, 1.16, 1.22, 0.93, 1.11, 0.76, 0.92, 0.88], [1.08, 1.26, 1.07, 0.97, 1.56, 0.77, 0.84, 1.22], [1.34, 1.54, 1.03, 0.92, 1.52, 0.9, 1.01, 1.34]])

# cash flow matrix
cf_matrix = np.zeros((4, 8))

# parameters
K = 1.1
rf = 0.06
paths = 8
T = 3

# cash flow matrix time 3
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
    regression = np.polyfit(X[T-t], Y[T-t], 2)
    # first is coefficient for X^2, second is coefficient X, third is constant
    print(regression)
    beta_2 = regression[0]
    slope = regression[1]
    intercept = regression[2]

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
    for i in range(paths): # todo: mistakes here
        if price_matrix[T-t, i] < K:
            # cont > ex --> t=3 is cf exercise, t=2 --> 0
            if continuation_value[0, i] > max(0, (K-price_matrix[T-t, i])):
                cf_matrix[T-t+1, i] = max(0, K-price_matrix[T-t+1, i])
                cf_matrix[T-t, i] = 0
            # cont < ex --> t=3 is 0, t=2 immediate exercise
            elif continuation_value[0, i] <= max(0, K-price_matrix[T-t, i]):
                cf_matrix[T-t, i] = max(0, K-price_matrix[T-t, i])
                cf_matrix[T-t+1, i] = 0
        # out of the money in t=2, t=2/3 both 0
        else:
            cf_matrix[T, i] = 0
            cf_matrix[T-t, i] = 0
