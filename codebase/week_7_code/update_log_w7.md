# Update Summary
## Implementing excitatory and inhibitory neurons
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
### A depends on presynaptic neuron
A takes two values according to excitatory or inhibitory neurons. But A is always positive.


## Implementing Short Term Plasticity

## New problems
### use abs or Relu for constraints?
Relu more solid to apply constraints, but clearing negative values to zero may slow down training.
We used abs for now.

### How would the tasks be used to examine working memory behaviour?


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

### Is optimisation algorithms like GA going the help the RNN structure?
GA can differ in performances from gradient basedd methods in many ways, and it can be used for parameter optimisation or optimisation of the whole structure. It also gives biological insights which in some senses 
agrees with bRNN. For example for some parameters like values for scaling matrix $A$ and time increment $dt$ we can use GA to decide its value rather than trained using gradient optimisation. But it might have problems with complexity and 
biological interpretations.
