import numpy as np

theta = 0.01
t=1
variance = (np.exp(2*theta*t)-1)
print(variance)
print(np.random.normal(0, variance))