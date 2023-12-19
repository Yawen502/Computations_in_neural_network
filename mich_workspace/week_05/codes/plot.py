import numpy as np
import matplotlib.pyplot as plt

constant = np.load('week_5_code/01_constant.npy')
conductance = np.load('week_5_code/02_conductance.npy')
simple_GRU = np.load('week_5_code/03_simple_GRU.npy')
con_small = np.load('week_5_code/01_constant_small.npy')
cond_small = np.load('week_5_code/02_conductance_small.npy')
gru_small = np.load('week_5_code/03_simple_GRU_small.npy')
con_hidsmall = np.load('week_5_code/01_constant_hidsmall.npy')
cond_hidsmall = np.load('week_5_code/02_conductance_hidsmall.npy')
gru_hidsmall = np.load('week_5_code/03_simple_GRU_hidsmall.npy')

x_values = np.arange(len(constant))
x_values2 = np.arange(len(constant)//2)

plt.figure(figsize = (12, 6))
#plt.plot(x_values, constant, label = 'Constant with input size 96, hidden size 100')
#plt.plot(x_values, conductance, label = 'Conductance with input size 96, hidden size 100')
#plt.plot(x_values, simple_GRU, label = 'Simple GRU with input size 96, hidden size 100')

#plt.plot(x_values, con_small, label = 'Constant with input size 48, hidden size 100')
#plt.plot(x_values, cond_small, label = 'Conductance with input size 48, hidden size 100')
#plt.plot(x_values, gru_small, label = 'Simple GRU with input size 48, hidden size 100')

plt.plot(x_values, con_hidsmall, label = 'Constant with input size 96, hidden size 50')
plt.plot(x_values2, cond_hidsmall, label = 'Conductance with input size 96, hidden size 50')
plt.plot(x_values, gru_hidsmall, label = 'Simple GRU with input size 96, hidden size 50')

plt.xlabel('Steps (in 100s)')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy over steps')

plt.legend()
plt.grid(True)
plt.show()