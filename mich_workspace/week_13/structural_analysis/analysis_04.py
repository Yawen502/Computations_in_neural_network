# read from pth file
import torch

# Specify the path to your .pth file
pth_file = '04_Dale-CB.pth'

# Load the model or tensor from the .pth file
data = torch.load(pth_file)
W = data['W']
P = data['P']
read_out = data['read_out']
#Accuracy of the model:82.3%

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
plt.title('Input vs Output for Dale-CB, Accuracy:82.3%')
plt.show()
