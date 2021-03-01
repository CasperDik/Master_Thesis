from LSMC.LSMC_American_option_quicker import LSMC, GBM
from Graphs.european_LSMC_vs_analytical import BSM
import numpy as np
import matplotlib.pyplot as plt

def plot_volatility_LSMC(S_0, K, T, dt, mu, rf, sigma, paths):
    LSMC_call = []
    LSMC_put = []

    for sigma in np.linspace(0, sigma*2, 20):
        for type in ["put", "call"]:
            if type == "call":
                price_matrix = GBM(T, dt, paths, mu, sigma, S_0)
                LSMC_call.append(LSMC(price_matrix, K, rf, paths, T, dt, type))
            elif type == "put":
                price_matrix = GBM(T, dt, paths, mu, sigma, S_0)
                LSMC_put.append(LSMC(price_matrix, K, rf, paths, T, dt, type))


    plt.plot(np.linspace(0, sigma*2, 20), LSMC_call, "--", label="LSMC call")
    plt.plot(np.linspace(0, sigma*2, 20), LSMC_put, "--", label="LSMC put")

    plt.legend()
    plt.title("Volatility vs option value - LSMC")
    plt.xlabel("Volatility")
    plt.ylabel("Option value")
    plt.show()

def plot_strike_LSMC(S_0, K, T, dt, mu, rf, sigma, paths):
    price_matrix = GBM(T, dt, paths, mu, sigma, S_0)
    LSMC_call = []
    LSMC_put = []

    for K in np.linspace(K - K / 4, K + K / 4, 20):
        for type in ["put", "call"]:
            if type == "call":
                LSMC_call.append(LSMC(price_matrix, K, rf, paths, T, dt, type))
            elif type == "put":
                LSMC_put.append(LSMC(price_matrix, K, rf, paths, T, dt, type))

    plt.plot(np.linspace(K-K/4, K+K/4, 20), LSMC_call, "--", label="LSMC call")
    plt.plot(np.linspace(K-K/4, K+K/4, 20), LSMC_put, "--", label="LSMC put")

    plt.legend()
    plt.title("Strike price vs option value - LSMC")
    plt.xlabel("Strike price")
    plt.ylabel("Option value")
    plt.show()

def plot_price_LSMC(S_0, K, T, dt, mu, rf, sigma, paths):
    LSMC_call = []
    LSMC_put = []

    for S in np.linspace(S_0 * 0.8, S_0 * 1.2, 20):
        for type in ["put", "call"]:
            if type == "call":
                price_matrix = GBM(T, dt, paths, mu, sigma, S)
                LSMC_call.append(LSMC(price_matrix, K, rf, paths, T, dt, type))
            elif type == "put":
                price_matrix = GBM(T, dt, paths, mu, sigma, S)
                LSMC_put.append(LSMC(price_matrix, K, rf, paths, T, dt, type))

    plt.plot(np.linspace(S_0 * 0.8, S_0 * 1.2, 20), LSMC_call, "--", label="LSMC call")
    plt.plot(np.linspace(S_0 * 0.8, S_0 * 1.2, 20), LSMC_put, "--", label="LSMC put")

    plt.legend()
    plt.title("Value of the option - LSMC")
    plt.xlabel("Asset price, St")
    plt.ylabel("Option value")
    plt.show()

def plot_maturity_LSMC(S_0, K, T, dt, mu, rf, sigma, paths):
    LSMC_call = []
    LSMC_put = []

    for time in np.linspace(T, T * 4, 20, dtype= int):
        for type in ["put", "call"]:
            if type == "call":
                price_matrix = GBM(time, dt, paths, mu, sigma, S_0)
                LSMC_call.append(LSMC(price_matrix, K, rf, paths, time, dt, type))
            elif type == "put":
                price_matrix = GBM(time, dt, paths, mu, sigma, S_0)
                LSMC_put.append(LSMC(price_matrix, K, rf, paths, time, dt, type))

    plt.plot(np.linspace(0, T * 4, 20), LSMC_call, "--", label="LSMC call")
    plt.plot(np.linspace(0, T * 4, 20), LSMC_put, "--", label="LSMC put")

    plt.legend()
    plt.title("Time to maturity vs option value - LSMC")
    plt.xlabel("Maturity")
    plt.ylabel("Option value")
    plt.show()

def american_vs_european(S_0, K, T, dt, mu, rf, sigma, paths):
    LSMC_call = []
    LSMC_put = []
    BSM_call = []
    BSM_put = []

    for S in np.linspace(S_0 * 0.8, S_0 * 1.2, 20):
        for type in ["put", "call"]:
            if type == "call":
                price_matrix = GBM(T, dt, paths, mu, sigma, S)
                LSMC_call.append(LSMC(price_matrix, K, rf, paths, T, dt, type))
            elif type == "put":
                price_matrix = GBM(T, dt, paths, mu, sigma, S)
                LSMC_put.append(LSMC(price_matrix, K, rf, paths, T, dt, type))
        call, put = BSM(S, K, rf, q, sigma, T)
        BSM_put.append(put)
        BSM_call.append(call)

    plt.plot(np.linspace(S_0 - S_0 / 2, S_0 + S_0 / 2, 20), LSMC_call, "--", label="LSMC call")
    plt.plot(np.linspace(S_0 - S_0 / 2, S_0 + S_0 / 2, 20), BSM_call, label="BSM call")
    plt.plot(np.linspace(S_0 - S_0 / 2, S_0 + S_0 / 2, 20), LSMC_put, "--", label="LSMC put")
    plt.plot(np.linspace(S_0 - S_0 / 2, S_0 + S_0 / 2, 20), BSM_put, label="BSM put")

    plt.legend()
    plt.title("European vs American option")
    plt.ylabel("Option value")
    plt.xlabel("Asset price, St")
    plt.show()

# inputs
paths = 5000


# years
T = 1
# execute possibilities per year
# american option large dt
dt = 365

K = 13
S_0 = 10
rf = 0.06
sigma = 0.4
r = 0.06
q = 0.00
mu = r - q

# plot_volatility_LSMC(S_0, K, T, dt, mu, rf, sigma, paths)
# plot_strike_LSMC(S_0, K, T, dt, mu, rf, sigma, paths)
# plot_price_LSMC(S_0, K, T, dt, mu, rf, sigma, paths)
# plot_maturity_LSMC(S_0, K, T, dt, mu, rf, sigma, paths)
american_vs_european(S_0, K, T, dt, mu, rf, sigma, paths)
