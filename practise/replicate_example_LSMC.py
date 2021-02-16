import numpy as np

""" 
Here I have replicated the numerical example in chapter 1 of the paper by Longstaff and Schwartz(2001) 
"""

# stock price paths matrix
# 8 paths, from t=0 to t=3
price_matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1.09, 1.16, 1.22, 0.93, 1.11, 0.76, 0.92, 0.88], [1.08, 1.26, 1.07, 0.97, 1.56, 0.77, 0.84, 1.22], [1.34, 1.54, 1.03, 0.92, 1.52, 0.9, 1.01, 1.34]])

# cash flow matrix
cf_matrix = np.zeros((4,8))

# parameters
K = 1.1
rf = 0.06

# cash flow matrix time 3
for p in range(8):
    cf_matrix[3, p] = max(0, K - (price_matrix[3, p]))

# find continuation value
# X = price in time T-1, Y = pv cf
Y = cf_matrix
X = price_matrix

for i in range(8):
    Y[2, i] = cf_matrix[3, i] * np.exp(-rf)

for j in range(7, -1, -1):
    if Y[3,j] == 0:
        Y = np.delete(Y,j, axis=1)
        X = np.delete(X, j, axis=1)

# regress Y on constant, X, X^2
regression = np.polyfit(X[2], Y[2], 2)
print(regression)


# compare immediate exercise with continuation value