Let's start with the Hodgkin-Huxley model,

<img width="319" alt="image" src="https://github.com/Yawen502/Computations_in_neural_network_Preparatorywork/assets/71087503/98432f9d-7e94-4586-b09d-0cb3b4bcf612">


We can then come up with the following equation. 
$$C^i \dot{V}^i(t) = - g_L^i+ g_E(t)(\mathcal{E}_E-V^i(t))+g_I^i(t)(\mathcal{E}_I-V^i(t))$$

Note the upper label $i$ represent the presynaptic cell, as the voltage depends on the presynaptic voltage signal. $g_L$ represents the leakage conductance, which is a constant. The reversal potential $\mathcal{E}_E$ and $\mathcal{E}_I$ are also constants which only depends on the type of postsynaptic cell. The other terms are all time dependent, as the model proposed.
we let z, the time constant term to be expressed as

$$z(t) = \frac{1}{C^i}(g_L^i+g_E^i+g_I^i)=G^i_b+ \frac{g_I(t)}{C^i}+\frac{g_E(t)}{C^i}$$

Then we reorganize to get
$$\dot{V}^i(t) = -z(t)^iV^i(t)+\frac{g_E^i}{C^i}\mathcal{E}_E-\frac{g_I^i}{C^i}\mathcal{E}_i$$

As the conductance of excitatory and inhibitory neurons are both directly proportional to the firing rate of neurons, we can introduce a matrix $K$ that intepretes this relationship:
$$\frac{g_E(t)}{C^i} = \sum_{i,j}^{}K_{ij}r_j$$
$$\frac{g_I(t)}{C^i} = \sum_{i,j}^{}K_{ij}r_j$$
