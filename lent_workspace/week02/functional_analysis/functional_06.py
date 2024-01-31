import matplotlib.pyplot as plt
import torch
import numpy as np
# Specify the path to your .pth file
pth_file = 'functional_06.pth'

# Load the model or tensor from the .pth file
data = torch.load(pth_file)
X_history = data['X_history']
U_history = data['U_history']
v_t_history = data['v_t_history']

# Assuming X_history is a list of tensors
X_history = [x.cpu().numpy() for x in X_history]
U_history = [u.cpu().numpy() for u in U_history]
v_t_history = [v.cpu().numpy() for v in v_t_history]

# check dimension of X_history
X_history = np.array(X_history)
U_history = np.array(U_history)
v_t_history = np.array(v_t_history)

# squeeze the dimension of X_history to (24, 100)
X_history = np.squeeze(X_history)
U_history = np.squeeze(U_history)
v_t_history = np.squeeze(v_t_history)

print(X_history.shape)
v_t_history = np.transpose(v_t_history)

# change device to cpu
def plot_dynamics(X_history, U_history, v_t_history):
    timesteps = np.arange(0, 100)
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(timesteps, X_history.T)
    plt.title('X Dynamics')
    plt.xlabel('Timestep')
    plt.ylabel('X Value')

    plt.subplot(1, 3, 2)
    plt.plot(timesteps, U_history.T)
    plt.title('U Dynamics')
    plt.xlabel('Timestep')
    plt.ylabel('U Value')

    plt.subplot(1, 3, 3)
    plt.plot(timesteps, v_t_history.T)
    plt.title('v_t Dynamics')
    plt.xlabel('Timestep')
    plt.ylabel('v_t Value')

    plt.tight_layout()
    plt.show()

# Plot first 3 neurons on the same plot
X_history = np.average(X_history, axis=0)
U_history = np.average(U_history, axis=0)
v_t_history = np.average(v_t_history, axis=0)

plot_dynamics(X_history, U_history, v_t_history)
