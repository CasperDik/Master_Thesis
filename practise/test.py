import numpy as np
import warnings
x = [1]
y = [2]


with warnings.catch_warnings():
    warnings.filterwarnings('error')
    try:
        coefficients = np.ma.polyfit(x, y, 2)
    except np.RankWarning:
        print("not enought data")