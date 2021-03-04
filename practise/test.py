import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x = np.linspace(1,10,11)
y = [1,3,4.9,7,8.5, 9.9, 10.8,12,12.6,13,13.1]
y = np.array(y, dtype=float)

def func(x,a,b):
    return a*np.log(x)+b

popt, pcov = curve_fit(func, x,y)
plt.plot(x,y)
plt.plot(sorted(x),func(sorted(x),*popt), "r")
plt.show()
