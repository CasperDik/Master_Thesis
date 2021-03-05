import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from Graphs.Plots_LSMC import perpetual_american

T = 41
# execute possibilities per year
# american option large dt
dt = 160

K = 120
S_0 = 130
rf = 0.07
sigma = 0.15
r = 0.07
q = 0.01
mu = r - q

perpetual_american(K, S_0, q, r, sigma)

x = np.linspace(1,27,14)
"""
y = [1,3,4.9,7,8.5, 9.9, 10.8,12,12.6,13,13.1]
y = np.array(y, dtype=float)

def func(x,a,b):
    return a*np.log(x)+b

popt, pcov = curve_fit(func, x,y)
plt.plot(x,y)
plt.plot(sorted(x),func(sorted(x),*popt), "r")
plt.show()
"""
print(x)
