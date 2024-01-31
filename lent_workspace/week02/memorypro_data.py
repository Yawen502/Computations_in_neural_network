# Generate train dataset for the MemoryPro task
# respond om the same direction as the stimulus after a memory period
# context period: rule input select the task
# memory period: rule and fixation on, stimulus off
# stimulus period: 
# response period:

import numpy as np
import torch

def memorypro_dataset(number):
    # return an input vector with dimension (6, timesteps) and its target output with dimension (3, timestep)
    # determine time intervals
    gamma = 0.2
    t_context = int(np.ceil(np.random.uniform(300, 700)))
    t_stim = int(np.ceil(np.random.uniform(200, 1600)))
    t_memory = int(np.ceil(np.random.uniform(200, 1600)))
    t_response = int(np.ceil(np.random.uniform(300, 700)))
    total = t_context + t_stim + t_memory + t_response
    total = int(total)
    print(total)
    input_set = []
    target_set = []
    u0 = 0.1* np.sqrt(2/gamma)
    for i in range(number):
        # generate theta randomly from 0 to 2pi
        theta = np.random.uniform(0, 2*np.pi)
        print(theta)
        u_fix = np.concatenate((np.repeat(1.0, total-t_response), np.repeat(0.0, t_response)), axis = None).reshape(total, 1)
        u_mod1_sin = np.concatenate((np.repeat(0.0, t_context), np.repeat(np.sin(theta), t_stim), np.repeat(0.0, t_memory+t_response)))
        u_mod1_sin = u_mod1_sin.reshape(total, 1)
        u_mod1_cos = np.concatenate((np.repeat(0.0, t_context), np.repeat(np.cos(theta), t_stim), np.repeat(0.0, t_memory+t_response)))
        u_mod1_cos = u_mod1_cos.reshape(total, 1)
        u_mod1 = np.concatenate((u_mod1_sin, u_mod1_cos), axis = 1).reshape(total, 2)
        u_mod2 = np.zeros((total, 2))
        u_rule = np.zeros((total, 1))
        u_noise = u0 * np.random.randn(6, total)   
        u = np.concatenate((u_fix, u_mod1, u_mod2, u_rule), axis = 1).T 
        input_set.append(u)

        # then determine the target output
        z_fix = u_fix
        z_sin = np.concatenate((np.repeat(0.0, t_context+t_stim+t_memory), np.repeat(np.sin(theta),t_response)))
        z_sin = z_sin.reshape(total, 1)
        z_cos = np.concatenate((np.repeat(0.0, t_context+t_stim+t_memory), np.repeat(np.cos(theta),t_response)))
        z_cos = z_cos.reshape(total, 1)
        z = np.concatenate((z_fix, z_sin, z_cos), axis = 1).T
        target_set.append(z)
    input_set = np.array(input_set)
    target_set = np.array(target_set)
    return input_set, target_set


import matplotlib.pyplot as plt

x, y = memorypro_dataset(10000)
plt.plot(x[0][0])
plt.plot(x[0][1])
plt.plot(x[0][2])
plt.plot(x[0][5])
plt.legend(['fix', 'sin', 'cos', '0', '0', '0'])
plt.show()

plt.plot(y[0][0])
plt.plot(y[0][1])
plt.plot(y[0][2])

plt.legend(['fix', 'sin', 'cos'])
plt.show()