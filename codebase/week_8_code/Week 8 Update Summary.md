

# Week 8 Update Summary
## Code update
### Modify the conductance_bRNN model in the following aspects
- Use a relaxed abs value
- Set a scale for b to make the firing rate values biologically plausible
- Simplify the weight matrix definition
- Alter the equation to be voltage equation, and define r_t using @property
  
### Implement STP and propose STP_conductance models
We have two sorts of STP network: 
- Synapse-based STP (SSTP): independent STP at individual synapses (where $x$ and $u$ become $N\times N$ matrices),
- Presynaptic STP (PSTP): tied at synaptic projections of the same neuron (where $x$ and $u$ become $N$-dimensinoal vectors).

We can have two orthogonal properties, SSTP and PSTP; and current-based model and conductance-based model. This week I implemented conductance-based model with SSTP and PSTP. Let's call this conductance_SSTP and conductance_PSTP bRNN model.


Conductance-based model written for SSTP(the matrix case):

$\boldsymbol{V}_{t+1} = (1-\boldsymbol{z}_t)\odot \boldsymbol{V}_t + \delta t(\boldsymbol{u}_t\odot \boldsymbol{x}_t \odot W)\boldsymbol{r}_t +\delta tP\boldsymbol{x}_t$

$\boldsymbol{r}_t = \phi(\boldsymbol{V}_t)$

$\boldsymbol{x}_{t+1} = z_x + (1-z_x)\boldsymbol{x}_t - \delta t \odot \boldsymbol{u}_t\odot \boldsymbol{x}_t\odot \boldsymbol{r}_t$

$\boldsymbol{u}_{t+1}= U z_u + (1-z_u)\boldsymbol{u}_t + \delta tU\odot (1-\boldsymbol{u}_t) \odot \boldsymbol{r}_t$


Conductance-based model written for PSTP(the vector case):

Other equations stay the same, but we now have 

$\boldsymbol{V}_{t+1} = (1-\boldsymbol{z}_t)\odot \boldsymbol{V}_t  + \delta t \, W (\boldsymbol{u}_t\odot \boldsymbol{x}_t \odot \boldsymbol{r}_t) + \delta t P\boldsymbol{x_t}$

#### Progress
Implemented the code, dimension check done. However I got an OutOfMemoryError.

#### Next step
Check for improvement in memory storage.

Switch to SSH server to run the code. (Basic set up is already done.)


