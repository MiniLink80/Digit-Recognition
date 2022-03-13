import numpy as np

x = np.random.rand(3,2)
y = np.sum(x, axis=0, keepdims=True)

print(y.shape)