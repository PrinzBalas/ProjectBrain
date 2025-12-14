import numpy as np

layers = [1, 4, 4]
for i in range(len(layers) - 1):
    w = np.random.randn(layers[i], layers[i + 1]) * 0.5

print(w.shape)