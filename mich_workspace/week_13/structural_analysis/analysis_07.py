# read from pth file
import torch

# Specify the path to your .pth file
#pth_file = "Dale-CB-weights.pth"
#pth_file = 'sigmoid_CB-RNN-tied-weights.pth'
#Accuracy of the model:77.08%
#pth_file = '07_CB-RNN-tied-STP.pth'
pth_file = 'analysis_08.pth'

# Load the model or tensor from the .pth file
data = torch.load(pth_file)
W = data['Weight Matrix W']
P = data['Input Weight Matrix P']
read_out = data['Readout Weights']
Ucap = data['Ucap']
z_u = data['z_u']
z_x = data['z_x']
# Accuracy of the model:61.22%

### Plot Input ratio ###
# calculate l2 norm of each neuron for P
import numpy as np
import matplotlib.pyplot as plt

input_strength = np.linalg.norm(P, axis=1)
# normalize
input_strength = input_strength / np.max(input_strength)

output_strength = np.linalg.norm(read_out, axis=0)
# normalize
output_strength = output_strength / np.max(output_strength)
# Add main title
plt.suptitle('Scatter Plot for CB-RNN-tied-STP')
plt.subplots(figsize=(5,5))
plt.subplot(2,2,1)
plt.scatter(input_strength, output_strength)
plt.xlabel('Input Strength')
plt.ylabel('Output Strength')
plt.title('Input Strength vs Output Strength')


input_ratio = input_strength / (input_strength + output_strength)
abs_W = np.abs(W)
normalization_factor = np.sum(abs_W, axis=1)
Ucap = Ucap.cpu().numpy()
z_u = z_u.cpu().numpy()
z_x = z_x.cpu().numpy()
Upost = np.sum(abs_W * Ucap, axis=0) / normalization_factor
z_x_post = np.sum(abs_W * z_x, axis=0) / normalization_factor
z_u_post = np.sum(abs_W * z_u, axis=0) / normalization_factor
plt.subplot(2,2,3)
plt.scatter(1/z_u, input_ratio)
plt.scatter(1/z_u_post, input_ratio)
plt.legend(['Pre-synaptic','Post-synaptic'])
plt.xlabel(r'$\tau_F$')
plt.ylabel('Input ratio')
plt.title(r'$\tau_F$ vs Input ratio')

plt.subplot(2,2,2)
plt.scatter(Ucap, input_ratio)
plt.scatter(Upost, input_ratio)
plt.legend(['Pre-synaptic','Post-synaptic'])
plt.xlabel('Upost')
plt.ylabel('Input ratio')
plt.title('Ucap vs Input ratio')

plt.subplot(2,2,4)
plt.scatter(1/z_x, input_ratio)
plt.scatter(1/z_x_post, input_ratio)
plt.legend(['Pre-synaptic','Post-synaptic'])
plt.xlabel(r'$\tau_D$')
plt.ylabel('Input ratio')
plt.title(r'$\tau_D$ vs Input ratio')
plt.tight_layout()
plt.show()

# Plot histogram of Input ratio, z_u, Ucap, z_x
plt.figure( figsize=(5,5))
# Large font
plt.subplot(2,2,1)
plt.hist(input_ratio, bins=30, edgecolor='white')
plt.xlabel('Input ratio')
plt.ylabel('count')
plt.title('Input ratio Histogram')

plt.subplot(2,2,2)
plt.hist(1/z_u, bins=30, edgecolor='white')
plt.xlabel(r'$\tau_F$')
plt.ylabel('count')
plt.title(r'$\tau_F$ Histogram')

plt.subplot(2,2,3)
plt.hist(Ucap, bins=30, edgecolor='white')
plt.xlabel('Ucap')
plt.ylabel('count')
plt.title('Ucap Histogram')

plt.subplot(2,2,4)
plt.hist(1/z_x, bins=30, edgecolor='white')
plt.xlabel(r'$\tau_D$')
plt.ylabel('count')
plt.title(r'$\tau_D$ Histogram')

plt.tight_layout()
plt.show()

# Plot a heatmap of W
plt.figure( figsize=(7,7))
# Large font
plt.imshow(W, cmap='bwr', vmin=-1, vmax=1)
plt.colorbar()
plt.xlabel('Pre-synaptic')
plt.ylabel('Post-synaptic')
plt.title('W')
plt.show()