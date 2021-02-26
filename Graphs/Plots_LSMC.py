from LSMC.LSMC_American_option_quicker import LSMC, GBM
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

    for K in np.linspace(K - K / 2, K + K / 2, 20):
        for type in ["put", "call"]:
            if type == "call":
                LSMC_call.append(LSMC(price_matrix, K, rf, paths, T, dt, type))
            elif type == "put":
                LSMC_put.append(LSMC(price_matrix, K, rf, paths, T, dt, type))

    plt.plot(np.linspace(K-K/2, K+K/2, 20), LSMC_call, "--", label="LSMC call")
    plt.plot(np.linspace(K-K/2, K+K/2, 20), LSMC_put, "--", label="LSMC put")

    plt.legend()
    plt.title("Volatility vs option value - LSMC")
    plt.xlabel("Strike price")
    plt.ylabel("Option value")
    plt.show()

def plot_price_LSMC(S_0, K, T, dt, mu, rf, sigma, paths):
    LSMC_call = []
    LSMC_put = []

    for S in np.linspace(S_0 - S_0 / 2, S_0 + S_0 / 2, 20):
        for type in ["put", "call"]:
            if type == "call":
                price_matrix = GBM(T, dt, paths, mu, sigma, S)
                LSMC_call.append(LSMC(price_matrix, K, rf, paths, T, dt, type))
            elif type == "put":
                price_matrix = GBM(T, dt, paths, mu, sigma, S)
                LSMC_put.append(LSMC(price_matrix, K, rf, paths, T, dt, type))

    plt.plot(np.linspace(S_0 - S_0 / 2, S_0 + S_0 / 2, 20), LSMC_call, "--", label="LSMC call")
    plt.plot(np.linspace(S_0 - S_0 / 2, S_0 + S_0 / 2, 20), LSMC_put, "--", label="LSMC put")

    plt.legend()
    plt.title("Value of the option - LSMC")
    plt.xlabel("Asset price, S")
    plt.ylabel("Option value")
    plt.show()

def plot_maturity_LSMC(S_0, K, T, dt, mu, rf, sigma, paths):
    LSMC_call = []
    LSMC_put = []

    for time in np.linspace(0, T * 2, 20):
        for type in ["put", "call"]:
            if type == "call":
                price_matrix = GBM(time, dt, paths, mu, sigma, S_0)
                LSMC_call.append(LSMC(price_matrix, K, rf, paths, time, dt, type))
            elif type == "put":
                price_matrix = GBM(time, dt, paths, mu, sigma, S_0)
                LSMC_put.append(LSMC(price_matrix, K, rf, paths, time, dt, type))

    plt.plot(np.linspace(0, T * 2, 20), LSMC_call, "--", label="LSMC call")
    plt.plot(np.linspace(0, T * 2, 20), LSMC_put, "--", label="LSMC put")

    plt.legend()
    plt.title("Time to maturity vs option value - LSMC")
    plt.xlabel("Maturity")
    plt.ylabel("Option value")
    plt.show()

# inputs
paths = 750
# years
T = 5
# execute possibilities per year
# american option large dt
dt = 365

K = 10
S_0 = 10
rf = 0.06
sigma = 0.2
mu = 0.06

plot_volatility_LSMC(S_0, K, T, dt, mu, rf, sigma, paths)
plot_strike_LSMC(S_0, K, T, dt, mu, rf, sigma, paths)
plot_price_LSMC(S_0, K, T, dt, mu, rf, sigma, paths)
plot_price_LSMC(S_0, K, T, dt, mu, rf, sigma, paths)