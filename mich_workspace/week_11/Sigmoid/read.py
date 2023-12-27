# read from pth file
import torch

# Specify the path to your .pth file
#pth_file = "Dale-CB-weights.pth"
#pth_file = 'sigmoid_CB-RNN-tied-weights.pth'
pth_file = 'relu_CB-RNN-tied-weights.pth'

# Load the model or tensor from the .pth file
data = torch.load(pth_file)
"""torch.save({
    'Weight Matrix W': W,
    'Input Weight Matrix P': P,
    'Readout Weights': read_out,
},'Dale-CB-weights.pth')"""

# calculate l2 norm of each neuron for P
import numpy as np
import matplotlib.pyplot as plt
input_strength = data['Input Weight Matrix P']
input_strength = np.linalg.norm(input_strength, axis=1)
# normalize
input_strength = input_strength / np.max(input_strength)

output_strength = data['Readout Weights']
output_strength = np.linalg.norm(output_strength, axis=0)
# normalize
output_strength = output_strength / np.max(output_strength)

plt.scatter(input_strength, output_strength)
plt.xlabel('Input Strength')
plt.ylabel('Output Strength')
plt.title('Input Strength vs Output Strength')
plt.show()