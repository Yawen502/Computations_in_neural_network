import numpy as np
import matplotlib.pyplot as plt

a1 = np.load('input_4_gru.npy')

x_values = np.arange(len(a1))

plt.figure(figsize = (12, 6))

plt.plot(x_values, a1, label = 'GRU with input 12')

plt.xlabel('Steps (in 100s)')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy over steps')

plt.legend()
plt.grid(True)
plt.show()