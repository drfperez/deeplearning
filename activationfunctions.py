import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0,x)

def leaky_relu(x):
    return np.maximum(0.1*x,x)

def elu(x):
    return np.where(x > 0, x, 1.0 * (np.exp(x) - 1))

x = np.linspace(-10,10,100)

plt.plot(x,sigmoid(x), label='Sigmoid')
plt.plot(x,tanh(x), label='Tanh')
plt.plot(x,relu(x), label='ReLU')
plt.plot(x,leaky_relu(x), label='Leaky ReLU')
plt.plot(x,elu(x), label='ELU')

plt.legend()
plt.show()
