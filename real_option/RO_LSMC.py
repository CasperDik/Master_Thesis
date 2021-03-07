import numpy as np
import time


def GBM(T, dt, paths, mu, sigma, S_0):
    # start timer
    tic = time.time()
    N = T * dt
    N = int(N)
    dt = 1 / dt

    price_matrix = np.exp((mu - sigma ** 2 / 2) * dt + sigma * np.random.normal(0, np.sqrt(dt), size=(paths, N)).T)
    price_matrix = np.vstack([np.ones(paths), price_matrix])
    price_matrix = S_0 * price_matrix.cumprod(axis=0)

    # Time and print the elapsed time
    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time of GBM: {:.2f} seconds'.format(elapsed_time))

    return price_matrix


def LSMC_RO(price_matrix, r, paths, T, dt, A, Q, epsilon, OPEX, mu, I):
    # start timer
    tic = time.time()

    # total number of steps
    N = T * dt
    N = int(N)

    # adjust yearly discount factor
    r = (1 + r) ** (1 / dt) - 1

    # cash flow matrix
    cf_matrix = np.zeros((N + 1, paths))

    # calculated cf when executed in time T (cfs European option)
    cf_matrix[N] = payoff_executing_RO(price_matrix[N], A, Q, epsilon, OPEX, r, mu, I, T)

    # 1 if in the money, otherwise 0
    execute = np.where(payoff_executing_RO(price_matrix, A, Q, epsilon, OPEX, r, mu, I, T) > 0, 1, 0)

    # consider all paths?
    # execute = np.ones_like(execute)

    for t in range(1, N):
        # discounted cf 1 time period
        discounted_cf = cf_matrix[N - t + 1] * np.exp(-r)

        # slice matrix and make all out of the money paths = 0 by multiplying with matrix "execute"
        X = price_matrix[N - t, :] * execute[N - t, :]

        # +1 here because otherwise will loose an in the money path at T-t,
        # that is out of the money in T-t+1(and thus has payoff=0)
        Y = (discounted_cf+1) * execute[N - t, :]

        # mask all zero values(out of the money paths) and run regression
        X1 = np.ma.masked_less_equal(X, 0)
        Y1 = np.ma.masked_less_equal(Y, 0) - 1

        # meaning all paths are out of the money, never optimal to exercise
        if X1.count() > 0:
            regression = np.ma.polyfit(X1, Y1, 2)
            # warnings.simplefilter('ignore', np.RankWarning)

            # calculate continuation value
            cont_value = np.zeros_like(Y1)
            cont_value = np.polyval(regression, X1)

            # update cash flow matrix
            imm_ex = payoff_executing_RO(X1, A, Q, epsilon, OPEX, r, mu, I, T)
            cf_matrix[N - t] = np.ma.where(imm_ex > cont_value, imm_ex, cf_matrix[N - t + 1] * np.exp(-r))
            cf_matrix[N - t + 1:] = np.ma.where(imm_ex > cont_value, 0, cf_matrix[N - t + 1:])
        else:
            cf_matrix[N - t] = cf_matrix[N - t + 1] * np.exp(-r)

    # obtain option value
    cf_matrix[0] = cf_matrix[1] * np.exp(-r)
    option_value = np.sum(cf_matrix[0]) / paths
    # st_dev = np.std(cf_matrix[0][cf_matrix[0] != 0])

    # Time and print the elapsed time
    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time of LSMC: {:.2f} seconds'.format(elapsed_time))

    print("Value of this option is:", option_value)
    print("Ran this with T: ", T, " and dt: ", dt)

    return option_value


def payoff_executing_RO(price, A, Q, epsilon, OPEX, r, mu, I, T):
    # discount factors continuous compounding
    df_ann = (1-np.exp(-r*T_plant))/(np.exp(r) - 1)
    df_growing_ann = (1 - np.exp((mu-r)*T_plant))/(np.exp(r-mu) - 1)

    NPV = (A * Q) * df_ann - (price * (1 / epsilon) * Q) * df_growing_ann - OPEX * df_ann - I
    return NPV.clip(min=0)


# inputs

# electricity price
A = 40
# Quantity per year
Q = 40000
# efficiency rate of the plant
epsilon = 0.85
# maintenance and operating cost per year
OPEX = 40000
# initial investment
I = 1400000
# discount rate (WACC?)
r = 0.06

# initial gas price
S_0 = 20
# drift rate mu of gas price
mu = 0.02
# volatility of the gas price
sigma = 0.15

# life of the power plant(in years)
T_plant = 30
# life of the option(in years)
T = 5
# time periods per year
dt = 365

# number of paths per simulations
paths = 1000

price_matrix = GBM(T, dt, paths, mu, sigma, S_0)
value = LSMC_RO(price_matrix, r, paths, T, dt, A, Q, epsilon, OPEX, mu, I)