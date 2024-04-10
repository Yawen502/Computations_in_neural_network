## week 11 Update
### Integrate flip flop task
Explanation of hyperparameters are added to overleaf. 

Hyperparameters of data generation: 

$t_\text{window}$: Longest time difference for two pulses to be counted as a `event'

$t_\text{total}$:total time steps

$t_\text{relax}$: number of time steps before output is reset  to 0

$P_\text{emit}$: this decides the sparsity of the input pulses we generate, when it is large, pulses are denser and more events happen for each channel

$n_\text{channel}$: number of flip flop channels

Hyperparameters of model training:

Batch size, learning rate, optimiser, number of neuron.

*Results: After tuning the hyperparameter to make the task simpler, it is still too hard for all the networks to do.*

![image](https://github.com/Yawen502/Computations_in_neural_network/assets/71087503/33ff643b-7413-452b-bcf4-9ffd0b179127)
MSE loss: 2.62e-01

For example lstm had this result when number of neuron = 30, batch size = 128, p_emit=0.05, t_window = 5, t_relax = 100
### time gap in snake MNIST:
*use time gap of 24, 48, 96 — does not drag down the performance of multiscale RNN*

### Pushing up network performance:
increasing number of neuron to 100 or 150 push the network performance to 85%

### Implementing behavioural flexibility task
Not done - do it next week

