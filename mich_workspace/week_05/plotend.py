
import numpy as np
import matplotlib.pyplot as plt

# Load the npy files
input = ['hidden size 50', 'hidden size 100']
a1 = np.load('week05/01_constant_hidsmall.npy')
a2 = np.load('week05/01_constant.npy')
b1 = np.load('week05/02_conductance_hidsmall.npy')
b2 = np.load('week05/02_conductance.npy')
c1 = np.load('week05/03_simple_GRU_hidsmall.npy')
c2 = np.load('week05/03_simple_GRU.npy')

# Extract the end points of each npy file
end1 = a1[-1]
end2 = a2[-1]
end3 = b1[-1]
end4 = b2[-1]
end5 = c1[-1]
end6 = c2[-1]

#reshape to enable concaatenate
end1 = end1.reshape(1, 1)
end2 = end2.reshape(1, 1)
end3 = end3.reshape(1, 1)
end4 = end4.reshape(1, 1)
end5 = end5.reshape(1, 1)
end6 = end6.reshape(1, 1)
# combine end points
# combine end1 and end2
end12 = np.concatenate((end1, end2), axis=0)
# combine end3 and end4
end34 = np.concatenate((end3, end4), axis=0)
# combine end5 and end6
end56 = np.concatenate((end5, end6), axis=0)
# Plot the end points of each npy file
plt.figure(figsize=(10, 6))
plt.plot(input, end12, 'o-', label='constant-A bRNN')
plt.plot(input, end34, 'o-', label='matrix-A bRNN')
plt.plot(input, end56, 'o-', label='simple-GRU')
# connect the points with lines
# Add a legend to the plot
plt.legend()

# Show the plot
plt.show()
