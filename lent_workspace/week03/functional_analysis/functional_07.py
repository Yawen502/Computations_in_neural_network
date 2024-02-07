import matplotlib.pyplot as plt
import torch
import numpy as np
# Specify the path to your .pth file
pth_file = 'functional_07.pth'

# Load the model or tensor from the .pth file
data = torch.load(pth_file)
X_history = data['X_history']
U_history = data['U_history']
v_t_history = data['v_t_history']
z_t_history = data['z_t_history']
# Just before saving...

# Convert a list of lists of tensors to a single numpy array
def convert_history(history_list):
    concatenated_history = []
    for batch in history_list:
        # Convert each tensor in the batch to numpy and append to a new list
        batch_history = [tensor.cpu().numpy() for tensor in batch]
        # Stack along a new dimension to keep batch data separate
        concatenated_history.append(np.stack(batch_history))
    # Concatenate all the batch data along the first dimension
    return np.concatenate(concatenated_history, axis=0)

# Apply this function to each history list
X  = convert_history(X_history)
U  = convert_history(U_history)
v_t  = convert_history(v_t_history)
v_t  = np.transpose(v_t , (0, 2, 1))
z_t  = convert_history(z_t_history)
# only take the first batch
X = X[:, :, 0]
U = U[:, :, 0]
v_t = v_t[:, :, 0]
z_t = z_t[:, :, 0]
print(X.shape)
# change device to cpu
# Plot scatter plot of the data

#line and point stype
plt.plot(np.arange(0, 196), X[:196, 0], label='X1')
plt.plot(np.arange(0, 196), X[:196, 1], label='X2')
plt.plot(np.arange(0, 196), X[:196, 2], label='X3')
plt.plot(np.arange(0, 196), X[:196, 3], label='X4')
plt.plot(np.arange(0, 196), X[:196, 4], label='X5')
plt.plot(np.arange(0, 196), X[:196, 5], label='X6')
plt.plot(np.arange(0, 196), U[:196, 6], label='U1')
plt.plot(np.arange(0, 196), U[:196, 7], label='U2')
plt.legend()
plt.xlabel('time')
plt.ylabel('X')
plt.title('X vs time')
plt.show()

plt.plot(np.arange(0, 196), U[:196, 0], label='U1')
plt.plot(np.arange(0, 196), U[:196, 1], label='U2')
plt.plot(np.arange(0, 196), U[:196, 2], label='U3')
plt.plot(np.arange(0, 196), U[:196, 3], label='U4')
plt.plot(np.arange(0, 196), U[:196, 4], label='U5')
plt.plot(np.arange(0, 196), U[:196, 5], label='U6')
plt.plot(np.arange(0, 196), U[:196, 6], label='U7')
plt.plot(np.arange(0, 196), U[:196, 7], label='U8')
plt.legend()
plt.xlabel('time')
plt.ylabel('U')
plt.title('U vs time')
plt.show()

plt.plot(np.arange(0, 196), v_t[:196, 0], label='v1')
plt.plot(np.arange(0, 196), v_t[:196, 1], label='v2')
plt.plot(np.arange(0, 196), v_t[:196, 2], label='v3')
plt.plot(np.arange(0, 196), v_t[:196, 3], label='v4')
plt.plot(np.arange(0, 196), v_t[:196, 4], label='v5')
plt.plot(np.arange(0, 196), v_t[:196, 5], label='v6')
plt.plot(np.arange(0, 196), v_t[:196, 6], label='v7')
plt.plot(np.arange(0, 196), v_t[:196, 7], label='v8')
plt.legend()
plt.xlabel('time')
plt.ylabel('v')
plt.title('v vs time')
plt.show()

plt.plot(np.arange(0, 196), z_t[:196, 0], label='z1')
plt.plot(np.arange(0, 196), z_t[:196, 1], label='z2')
plt.plot(np.arange(0, 196), z_t[:196, 2], label='z3')
plt.plot(np.arange(0, 196), z_t[:196, 3], label='z4')
plt.plot(np.arange(0, 196), z_t[:196, 4], label='z5')
plt.plot(np.arange(0, 196), z_t[:196, 5], label='z6')
plt.plot(np.arange(0, 196), z_t[:196, 6], label='z7')
plt.plot(np.arange(0, 196), z_t[:196, 7], label='z8')
plt.legend()
plt.xlabel('time')
plt.ylabel('z')
plt.title('z vs time')
plt.show()

# correlation between X and U
print(np.corrcoef(X[:, 0], U[:, 0]))
print(np.corrcoef(X[:, 1], U[:, 1]))
print(np.corrcoef(X[:, 2], U[:, 2]))