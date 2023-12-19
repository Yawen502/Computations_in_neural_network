# Update Summary
## Code modification: Implementing excitatory and inhibitory neurons
### Adding constraints to weight matrix $W$
We ensure that $W$ always satisfies Dale's principle:

    def forward(self, x):        
            with torch.no_grad():
                self.w_r.data[:self.hidden_size//2, :self.hidden_size//2] = torch.abs(self.w_r.data[:self.hidden_size//2, :self.hidden_size//2])
                self.w_r.data[self.hidden_size//2:, :self.hidden_size//2] = torch.abs(self.w_r.data[self.hidden_size//2:, :self.hidden_size//2])
                self.w_r.data[:self.hidden_size//2, self.hidden_size//2:] = -torch.abs(self.w_r.data[:self.hidden_size//2, self.hidden_size//2:])
                self.w_r.data[self.hidden_size//2:, self.hidden_size//2:] = -torch.abs(self.w_r.data[self.hidden_size//2:, self.hidden_size//2:])

### Modify firing rate to be always positive
Tanh is changed to Sigmoid to ensure $r_t$ is always positive.

        self.z_t = self.dt * self.Sigmoid(torch.matmul(w_z, self.A * self.r_t) + torch.matmul(p_z, x) + self.g_z)
        self.r_t = (1 - self.z_t) * self.r_t + self.z_t * self.Sigmoid(torch.matmul(self.w_r, self.r_t) + torch.matmul(self.p_r, x) + self.b_r)

### Number of neurons doubled up
We use n=200 (a relatively large number) for preliminary testing

### Scaling factor A depends on presynaptic neuron
A takes two values according to excitatory or inhibitory neurons. But A is always positive.

            class customGRUCell(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers):
                    ...
                    self.a_excitatory = nn.Parameter(torch.tensor(0.5), requires_grad=True)
                    self.a_inhibitory = nn.Parameter(torch.tensor(0.5), requires_grad=True)
                def forward(self, x):  
                    ...
                    is_excitatory = self.w_r.data > 0  
                    a = torch.where(is_excitatory, self.a_excitatory, self.a_inhibitory)
                    self.A = 10 * self.Sigmoid(a)
                    self.A = self.A.transpose(0, 1)
                    
                    self.z_t = self.dt * self.Sigmoid(torch.matmul(w_z, torch.matmul(self.A,self.r_t)) + torch.matmul(p_z, x) + self.g_z)

### Only excitatory neurons give outputs

        class customGRUCell(nn.Module):
            ...
            def forward(self, x): 
            ...
            # zero out inhibitory neurons for output
            excitatory_mask = self.w_r.data > 0  # Mask for excitatory cells
            excitatory_mask = excitatory_mask.any(dim=1).unsqueeze(0) # Match the shape of r_t
            self.excitatory_outputs = self.r_t * excitatory_mask


        class customGRU(nn.Module):
            ...
            def forward(self, x):
                if self.batch_first == True:
                    ...
                return self.rnncell.excitatory_outputs   
        
## Implementing Short Term Plasticity
STP model follows equation

$h_t = (1-z_h)h_{t-1}+z_h\odot \phi ((u_t\odot x_t \odot W)h_{t-1}+PX_t+b)$ where $\phi$ is Sigmoid() function. 

$x_t = z_x + (1-z_x)x_{t-1}-(\delta t\odot u_t\odot x_{t-1})h_{t-1}$

$u_t = Uz_u + (1-z_u)u_{t-1}+(\delta t\odot U\odot (1-u_{t-1}))h_{t-1}$

We have dynamic $u_t$, $x_t$. $z_h$, $z_u$ and $z_x$ are fixed, trained parameters, which is constrained within $[0, 1]$ and optimised during training.


For conductance model we have dynamic time constant $z_t$:

$z_t^{i} = \delta t \times \phi (G_b^i+ \sum_{j} A_{ij}\left|W_{ij}  \right| r_j + \left|P_{in}\right|x_n)$

We can take $z_t$ to preserve the feature of conductance-based model, while introducing the dynamically updated $u_t$ and $x_t$. We therefore have

$h_t = (1-z_t)h_{t-1}+z_t\odot \phi ((u_t\odot x_t \odot W)h_{t-1}+PX_t+b)$

This can be a possible way to combine these two models together.


## Further Investigation
### Use abs or Relu for constraints of Dale's principle?
Relu is more solid to apply constraints, but clearing negative values to zero may slow down training.
We used abs for now.

### How would the tasks be used to examine working memory behaviour?
- By using increasing number of sequence length (or we can say by using smaller input size), the task required longer working memory to memorize all the input sequences to perform the classification tasks. Therefore, the models' behaviour at very small input size is a strong evidence for the working memory of the model. Our results show that constant-A bRNN and matrix-A bRNN have much longer working memory than vanilla RNN, and they have comparable working memory to simple GRU.
- By testing the model further through tasks with extreme long sequence length,(and check the decay in accuracy rate) we can see the working memory behaviour even clearer.
- By tracking the change of $z_t$ and $r_t$ during training, we can observe the working memory behaviours: stable trajectories represent working memory and faster decay represent forgetting.


### Parameter tracking
Parameter examples:

      Weight matrix W:
      tensor([ 0.0990,  0.0495,  0.0396,  ..., -0.0413, -0.0279, -0.0756],
      [ 0.0210,  0.0721,  0.0679,  ..., -0.0639, -0.0296, -0.0509],
      [ 0.0650,  0.1060,  0.0857,  ..., -0.1080, -0.0914, -0.1100],)

      Firing rate r_t:
      tensor([[0.1056, 0.1912, 0.0916,  ..., 0.2911, 0.3263, 0.3394],
        [0.0820, 0.1569, 0.0608,  ..., 0.2924, 0.3375, 0.3561],
        [0.0838, 0.1566, 0.0626,  ..., 0.2959, 0.3384, 0.3542],
        ...,
        [0.0913, 0.1692, 0.0758,  ..., 0.2955, 0.3252, 0.3387],
        [0.1099, 0.1921, 0.0953,  ..., 0.2924, 0.3240, 0.3393],
        [0.0722, 0.1397, 0.0488,  ..., 0.3040, 0.3539, 0.3734]],

we discovered $w_r$ and $r_t$ are both very small, probably due to the constraints on scaling factors. Modifying that may improve the model performance. Check that!

### (a new idea from lectures) Is optimisation algorithms like GA going the help the RNN structure?
Genetic Algorithm can differ in performances from gradient basedd methods in many ways, and it can be used for parameter optimisation or optimisation of the whole structure. It also gives biological insights which in some senses 
agrees with bRNN. For example for some parameters like values for scaling matrix $A$ and time increment $dt$ we can use GA to decide its value rather than trained using gradient optimisation. But it might have problems with complexity and 
biological interpretations.


### Comparison of Model

Before applying excitatory and inhibitory neurons: Accuracy of the model:40.06%

After: Accuracy of the model:37.72%

### This week
Implement the STP model and the new conductance model with excitatory and inhibitory features. ADD them to the accuracy comparison that we had.
In the future: 
We can crop the images(after training for 25%, erase the hidden state), or just only feed 25%, 50% etc. data to the model. Check how the accuracies decay for different models.


