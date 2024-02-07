## Initialising Dale's Law

Repeat the initialisation until NSE is high enough (which indicates the spectral is concentrated enough, which is an indication of good test performance)

            while True:
                self.K = torch.nn.Parameter(self.init_dale(self.hidden_size, self.hidden_size))
                self.C = torch.nn.Parameter(self.init_dale(self.hidden_size, self.hidden_size))
                nse_K = self.NSE(self.K)
                nse_C = self.NSE(self.C)
                if nse_K > 0.905 and nse_C > 0.905:
                    break

            def NSE(self, weights):
                values = torch.linalg.svdvals(weights)
                normalised_v = values/sum(values)
                H = -1/torch.log(torch.tensor(self.hidden_size)) * torch.sum(normalised_v * torch.log(normalised_v))
                #print(H)
                return H

## MemoryPro task
I create the dataset according to the paper, which can generate arbitary size of dataset that follow the description.

![Untitled](https://github.com/Yawen502/Computations_in_neural_network/assets/71087503/35ef6a30-ed7c-43ed-a5a0-689bf5745804)


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

