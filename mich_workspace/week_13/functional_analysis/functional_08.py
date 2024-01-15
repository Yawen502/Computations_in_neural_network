import matplotlib.pyplot as plt
import torch
import numpy as np
# Specify the path to your .pth file
pth_file = 'functional_08.pth'

# Load the model or tensor from the .pth file
data = torch.load(pth_file)
weights = torch.load('analysis_08.pth')

X_history = data['X_history']
U_history = data['U_history']
v_t_history = data['v_t_history']
z_t_history = data['z_t_history']
W = weights['Weight Matrix W']


print(type(X_history))
print(len(X_history))
print(type(X_history[0]))
print(type(X_history[0][0]))

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

X  = np.mean(X , axis=0)
X  = np.mean(X , axis=1)
U  = np.mean(U , axis=0)
U  = np.mean(U , axis=1)
v_t  = np.mean(v_t , axis=0)
v_t  = np.mean(v_t , axis=1)
z_t  = np.mean(z_t , axis=0)
z_t  = np.mean(z_t , axis=1)

# change device to cpu
# Plot scatter plot of the data
plt.subplots(figsize=(7, 7))
plt.subplot(2, 2, 1)
plt.scatter(X , U )
plt.xlabel('X')
plt.ylabel('U')
plt.title('X vs U')

plt.subplot(2, 2, 2)
plt.scatter(X , v_t )
plt.xlabel('X')
plt.ylabel('v_t')
plt.title('X vs v_t')

plt.subplot(2, 2, 3)
plt.scatter(U , v_t )
plt.xlabel('U')
plt.ylabel('v_t')
plt.title('U vs v_t')

plt.subplot(2, 2, 4)
plt.scatter(1/z_t , v_t )
plt.xlabel('1/z_t')
plt.ylabel('v_t')
plt.title('1/z_t vs v_t')

plt.tight_layout()
plt.show()

# plot histogram of the four variables
plt.subplots(figsize=(7, 7))
plt.subplot(2, 2, 1)
plt.hist(X , bins=50)
plt.xlabel('X')
plt.ylabel('Count')
plt.title('X')

plt.subplot(2, 2, 2)
plt.hist(U , bins=50)
plt.xlabel('U')
plt.ylabel('Count')
plt.title('U')

plt.subplot(2, 2, 3)
plt.hist(v_t , bins=50)
plt.xlabel('v_t')
plt.ylabel('Count')
plt.title('v_t')

plt.subplot(2, 2, 4)
plt.hist(1/z_t , bins=50)
plt.xlabel('1/z_t')
plt.ylabel('Count')
plt.title('1/z_t')
plt.tight_layout()
plt.show()