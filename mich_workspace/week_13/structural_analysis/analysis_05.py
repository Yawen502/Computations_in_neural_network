# read from pth file
import torch

# Specify the path to your .pth file
#pth_file = "Dale-CB-weights.pth"
#pth_file = 'sigmoid_CB-RNN-tied-weights.pth'
pth_file = '05_CB-GRU-STP.pth'

# Load the model or tensor from the .pth file
data = torch.load(pth_file)
W = data['W']
P = data['P']
read_out = data['read_out']
Ucap = data['Ucap'].cpu().numpy()
z_u = data['z_u'].cpu().numpy()
z_x = data['z_x'].cpu().numpy()

### Plot input/output ratio ###
# calculate l2 norm of each neuron for P
import numpy as np
import matplotlib.pyplot as plt

input_strength = np.linalg.norm(P, axis=1)
# normalize
input_strength = input_strength / np.max(input_strength)

output_strength = np.linalg.norm(read_out, axis=0)
# normalize
output_strength = output_strength / np.max(output_strength)

plt.scatter(input_strength, output_strength)
plt.xlabel('Input Strength')
plt.ylabel('Output Strength')
plt.title('Input Strength vs Output Strength')
plt.show()

input_ratio = input_strength / (input_strength + output_strength)

abs_W = np.abs(W)
normalization_factor = np.sum(abs_W, axis=1)

Upost = np.sum(abs_W * Ucap, axis=0) / normalization_factor
z_x_post = np.sum(abs_W * z_x, axis=0) / normalization_factor
z_u_post = np.sum(abs_W * z_u, axis=0) / normalization_factor
plt.scatter(1 / z_u, input_ratio)
plt.scatter(1/ z_u_post, input_ratio)
plt.legend(['Pre-synaptic','Post-synaptic'])
plt.xlabel(r'$\tau_D$')
plt.ylabel('input/output ratio')
plt.title('1 / z_u vs IO Ratio')
plt.show()

plt.scatter(1-Ucap, input_ratio)
plt.scatter(Upost, input_ratio)
plt.legend(['Pre-synaptic','Post-synaptic'])
plt.xlabel('Upost')
plt.ylabel('input/output ratio')
plt.title('Ucap vs IO Ratio')
plt.show()

plt.scatter(1 / z_x, input_ratio)
plt.scatter(1 / z_x_post, input_ratio)
plt.legend(['Pre-synaptic','Post-synaptic'])
plt.xlabel(r'$\tau_F$')
plt.ylabel('input/output ratio')
plt.title('z_x vs IO Ratio')
plt.show()