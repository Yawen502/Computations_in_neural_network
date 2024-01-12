# read from pth file
import torch

# Specify the path to your .pth file
#pth_file = "Dale-CB-weights.pth"
#pth_file = 'sigmoid_CB-RNN-tied-weights.pth'
#Accuracy of the model:77.08%
pth_file = '07_CN-RNN-tied-STP.pth'

# Load the model or tensor from the .pth file
data = torch.load(pth_file)
W = data['W']
P = data['P']
read_out = data['read_out']
Ucap = data['Ucap']
z_u = data['z_u']
z_x = data['z_x']
# Accuracy of the model:61.22%

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
plt.rcParams.update({'font.size': 12})
# Add main title
plt.suptitle('Scatter Plot for CB-RNN-tied-STP')

plt.subplot(2,2,1)
plt.scatter(input_strength, output_strength)
plt.xlabel('Input Strength')
plt.ylabel('Output Strength')
plt.title('Input Strength vs Output Strength, Accuracy :61.22%')



abs_W = np.abs(W)
normalization_factor = np.sum(abs_W, axis=1)

Upost = np.sum(abs_W * Ucap, axis=0) / normalization_factor
z_x_post = np.sum(abs_W * z_x, axis=0) / normalization_factor
z_u_post = np.sum(abs_W * z_u, axis=0) / normalization_factor
plt.subplot(2,2,2)
plt.scatter(z_u, input_strength/output_strength)
plt.scatter(z_u_post, input_strength/output_strength)
plt.legend(['Pre-synaptic','Post-synaptic'])
plt.xlabel('z_u')
plt.ylabel('input/output ratio')
plt.title('z_u vs IO Ratio (post-synaptic)')

plt.subplot(2,2,3)
plt.scatter(Ucap, input_strength/output_strength)
plt.scatter(Upost, input_strength/output_strength)
plt.legend(['Pre-synaptic','Post-synaptic'])
plt.xlabel('Upost')
plt.ylabel('input/output ratio')
plt.title('Ucap vs IO Ratio')

plt.subplot(2,2,4)
plt.scatter(z_x, input_strength/output_strength)
plt.scatter(z_x_post, input_strength/output_strength)
plt.legend(['Pre-synaptic','Post-synaptic'])
plt.xlabel('z_x')
plt.ylabel('input/output ratio')
plt.title('z_x vs IO Ratio')
plt.tight_layout()
plt.show()

# Plot histogram of input/output ratio, z_u, Ucap, z_x
plt.figure(figsize=(10,10))
# Large font
plt.rcParams.update({'font.size': 15})
plt.subplot(2,2,1)
plt.hist(input_strength/output_strength, bins=30, edgecolor='white')
plt.xlabel('input/output ratio')
plt.ylabel('count')
plt.title('IO Ratio Histogram')

plt.subplot(2,2,2)
plt.hist(z_u, bins=30, edgecolor='white')
plt.xlabel('z_u')
plt.ylabel('count')
plt.title('z_u Histogram')

plt.subplot(2,2,3)
plt.hist(Ucap, bins=30, edgecolor='white')
plt.xlabel('Ucap')
plt.ylabel('count')
plt.title('Ucap Histogram')

plt.subplot(2,2,4)
plt.hist(z_x, bins=30, edgecolor='white')
plt.xlabel('z_x')
plt.ylabel('count')
plt.title('z_x Histogram')

plt.tight_layout()
plt.show()