import matplotlib.pyplot as plt
import torch
# Specify the path to your .pth file
pth_file = 'functional_05.pth'

# Load the model or tensor from the .pth file
data = torch.load(pth_file)
X_history = data['X_history']
U_history = data['U_history']
v_t_history = data['v_t_history']
print(X_history[0])

def plot_dynamics(X_history, U_history, v_t_history):
    timesteps = range(len(X_history))
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(timesteps, X_history)
    plt.title('X Dynamics')
    plt.xlabel('Timestep')
    plt.ylabel('X Value')

    plt.subplot(1, 3, 2)
    plt.plot(timesteps, U_history)
    plt.title('U Dynamics')
    plt.xlabel('Timestep')
    plt.ylabel('U Value')

    plt.subplot(1, 3, 3)
    plt.plot(timesteps, v_t_history)
    plt.title('v_t Dynamics')
    plt.xlabel('Timestep')
    plt.ylabel('v_t Value')

    plt.tight_layout()
    plt.show()

plot_dynamics(X_history[0], U_history[0], v_t_history[0])
plot_dynamics(X_history[1], U_history[1], v_t_history[1])
plot_dynamics(X_history[2], U_history[2], v_t_history[2])