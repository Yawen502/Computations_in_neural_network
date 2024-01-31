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
P = weights['Input Weight Matrix P']
read_out = weights['Readout Weights']
Ucap = weights['Ucap']
z_u = weights['z_u']
z_x = weights['z_x']

input_strength = np.linalg.norm(P, axis=1)
# normalize
input_strength = input_strength / np.max(input_strength)

output_strength = np.linalg.norm(read_out, axis=0)
# normalize
output_strength = output_strength / np.max(output_strength)

input_ratio = input_strength / (input_strength + output_strength)
abs_W = np.abs(W)
normalization_factor = np.sum(abs_W, axis=1)
Ucap = Ucap.cpu().numpy()
z_u = z_u.cpu().numpy()
z_x = z_x.cpu().numpy()
print(z_u.shape, z_x.shape, Ucap.shape)
# reshape to (48,)
Ucap = np.reshape(Ucap, (48,))
z_u = np.reshape(z_u, (48,))
z_x = np.reshape(z_x, (48,))
Upost = np.sum(abs_W * Ucap, axis=0) / normalization_factor
z_x_post = np.sum(abs_W * z_x, axis=0) / normalization_factor
z_u_post = np.sum(abs_W * z_u, axis=0) / normalization_factor

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
print(X.shape, U.shape, v_t.shape, z_t.shape)
import pandas as pd
import seaborn as sns


# Assuming X, U, v_t, and z_t are numpy arrays of the same length
data = {
    r'$x_{mean}$': X,
    r'$u_{mean}$': U,
    '$v_t$': v_t,
    '$z_t$': z_t,
    'Ucap': Ucap,
    r'$\tau_F$': z_u,
    r'$\tau_D$': z_x,
    'input_ratio': input_ratio
}
data2 = {
    'Ucap': Ucap,
    r'$\tau_F$': z_u,
    r'$\tau_D$': z_x,
    'input_ratio': input_ratio,
}
data3 = {
    r'$x_{mean}$': X,
    r'$u_{mean}$': U,
    '$v_t$': v_t,
    '$z_t$': z_t,
}
# Create a DataFrame
df = pd.DataFrame(data)
df2 = pd.DataFrame(data2)
df3 = pd.DataFrame(data3)
# Plotting using Seaborn
#set plot size
plt.figure(figsize=(7,7))
sns.set_theme(style="ticks")
#pair_plot = sns.pairplot(df3)  # Use hue='your_categorical_variable' if you have one
#plt.show()
#pair_plot2 = sns.pairplot(df2)  # Use hue='your_categorical_variable' if you have one
#plt.show()

# Sample DataFrame
# df is your existing pandas DataFrame with two columns, for example, 'X' and 'U'

# Creating a joint plot
joint_plot = sns.jointplot(data=df, x=r'$u_{mean}$', y=r'$x_{mean}$', kind='scatter', height=4)
#joint_plot = sns.jointplot(data=df, x=r'$u_{mean}$', y='Ucap', kind='scatter')
joint_plot = sns.jointplot(data=df, x='$z_t$', y='Ucap', kind='scatter', height=4)
# Display the plot
plt.show()

'''
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

# Plot a heatmap of W
plt.figure( figsize=(7,7))
# Large font
# find max and mean of W

plt.rcParams.update({'font.size': 15})
plt.imshow(W, cmap='bwr', vmin=np.min(W), vmax=np.max(W))
plt.colorbar()
plt.xlabel('Pre-synaptic')
plt.ylabel('Post-synaptic')
plt.title('W')
plt.show()
'''