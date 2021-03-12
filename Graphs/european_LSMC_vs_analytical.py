from LSMC.LSMC_American_option_faster import LSMC, GBM
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

"""Comparing the option values of european options from LSMC method with the analytical solutions from BSM"""

def BSM(S_0, K, r, q, sigma, T):
    d1 = (np.log(S_0/K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S_0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S_0 * np.exp(-q * T) * norm.cdf(-d1)
    return call, put

def plot_strike_GBMvsLSMC(S_0, K, T, dt, mu, r, q, sigma, paths):
    price_matrix = GBM(T, dt, paths, mu, sigma, S_0)
    LSMC_call = []
    LSMC_put = []
    BSM_call = []
    BSM_put = []

    for K in np.linspace(K - K / 2, K + K / 2, 20):
        for type in ["put", "call"]:
            if type == "call":
                LSMC_call.append(LSMC(price_matrix, K, r, paths, T, dt, type))
            elif type == "put":
                LSMC_put.append(LSMC(price_matrix, K, r, paths, T, dt, type))
        call, put = BSM(S_0, K, r, q, sigma, T)
        BSM_put.append(put)
        BSM_call.append(call)

    plt.plot(np.linspace(K-K/2, K+K/2, 20), LSMC_call, "--", label="LSMC call")
    plt.plot(np.linspace(K-K/2, K+K/2, 20), BSM_call, label="BSM call", alpha=0.7)
    plt.plot(np.linspace(K-K/2, K+K/2, 20), LSMC_put, "--", label="LSMC put")
    plt.plot(np.linspace(K-K/2, K+K/2, 20), BSM_put, label="BSM put", alpha=0.7)

    plt.legend()
    plt.title("Analytical solutions BSM vs LSMC of european option")
    plt.xlabel("Strike price")
    plt.ylabel("Option value")
    plt.show()

def plot_volatility_GBMvsLSMC(S_0, K, T, dt, mu, r, q, sigma, paths):
    LSMC_call = []
    LSMC_put = []
    BSM_call = []
    BSM_put = []

    for sigma in np.linspace(0, sigma*2, 20):
        for type in ["put", "call"]:
            if type == "call":
                price_matrix = GBM(T, dt, paths, mu, sigma, S_0)
                LSMC_call.append(LSMC(price_matrix, K, r, paths, T, dt, type))
            elif type == "put":
                price_matrix = GBM(T, dt, paths, mu, sigma, S_0)
                LSMC_put.append(LSMC(price_matrix, K, r, paths, T, dt, type))
        call, put = BSM(S_0, K, r, q, sigma, T)
        BSM_put.append(put)
        BSM_call.append(call)

    plt.plot(np.linspace(0, sigma*2, 20), LSMC_call, "--", label="LSMC call")
    plt.plot(np.linspace(0, sigma*2, 20), BSM_call, label="BSM call", alpha=0.7)
    plt.plot(np.linspace(0, sigma*2, 20), LSMC_put, "--", label="LSMC put")
    plt.plot(np.linspace(0, sigma*2, 20), BSM_put, label="BSM put", alpha=0.7)

    plt.legend()
    plt.title("Analytical solutions BSM vs LSMC of european option")
    plt.xlabel("Volatility")
    plt.ylabel("Option value")
    plt.show()


if __name__ == "__main__":
    paths = 200000
    # years
    T = 2
    # execute possibilities per year
    # has to be 1 otherwise not european option
    dt = 0.5

    K = 130
    S_0 = 130
    sigma = 0.2
    r = 0.07
    q = 0.01
    mu = r - q

    plot_strike_GBMvsLSMC(S_0, K, T, dt, mu, r, q, sigma, paths)
    plot_volatility_GBMvsLSMC(S_0, K, T, dt, mu, r, q, sigma, paths)
