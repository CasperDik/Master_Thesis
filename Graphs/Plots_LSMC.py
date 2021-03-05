from LSMC.LSMC_American_option_quicker import LSMC, GBM
from Graphs.european_LSMC_vs_analytical import BSM
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
        call, put = BSM(S_0, K, rf, q, sigma, T)
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

def perpetual_american(K, S_0, q, r, sigma):
    B1 = (q+0.5*sigma**2-r)/sigma**2 + (np.sqrt((r-q-0.5*sigma**2)**2 + 2*r*sigma**2))/sigma**2
    Sbar = B1/(B1-1) * K
    print((Sbar-K)*(S_0/Sbar)**B1)
    return (Sbar-K)*(S_0/Sbar)**B1

def convergence_american_perpetual(T, dt, paths, mu, sigma, S_0, type):
    T = T+1
    lsmc_call = []
    confidence_interval_up = []
    confidence_interval_down = []
    x = np.linspace(1, T, 14)

    for T in x:
        # slice = int(T*dt)
        price_matrix = GBM(T, dt, paths, mu, sigma, S_0)
        val, st_dev = LSMC(price_matrix, K, rf, paths, T, dt, type)
        lsmc_call.append(val)
        # confidence_interval_up.append(val + 1.96 * st_dev / np.sqrt(paths))
        # confidence_interval_down.append(val - 1.96 * st_dev / np.sqrt(paths))

    lsmc_call = np.array(lsmc_call, dtype=float)
    def func(x, a, b):
        return a * np.log(x) + b
    popt, pcov = curve_fit(func, x, lsmc_call)
    plt.plot(sorted(x), func(sorted(x), *popt), "k", linestyle="dashed")

    plt.plot(x, lsmc_call, "x", c="skyblue")
    # plt.fill_between(x, confidence_interval_up, confidence_interval_down, "b", alpha=0.1)
    plt.axhline(y=perpetual_american(K, S_0, q, r, sigma), c="r")
    plt.title("Convergence of the LSMC to perpetual American option")
    plt.xlabel("Years")
    plt.ylabel("Option value")
    plt.plot()
    plt.show()

# inputs
paths = 50000

# years
T = 26
# execute possibilities per year
# american option large dt
dt = 12

K = 120
S_0 = 130
rf = 0.07
sigma = 0.15
r = 0.07
q = 0.01
mu = r - q

# perpetual_american(K, S_0, q, r, sigma)

# plot_volatility_LSMC(S_0, K, T, dt, mu, rf, sigma, paths)
# plot_strike_LSMC(S_0, K, T, dt, mu, rf, sigma, paths)
# plot_price_LSMC(S_0, K, T, dt, mu, rf, sigma, paths)
# plot_maturity_LSMC(S_0, K, T, dt, mu, rf, sigma, paths)
# american_vs_european(S_0, K, T, dt, mu, rf, sigma, paths)
convergence_american_perpetual(T, dt, paths, mu, sigma, S_0, "call")
