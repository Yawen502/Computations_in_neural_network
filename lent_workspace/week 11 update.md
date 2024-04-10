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
### time gap in snake MNIST:
use time gap of 24, 48, 96 — does not drag down the performance of multiscale RNN


3. increasing number of neuron to 100 or 150 push the network performance to 85%
4. Not done - do it next week
5. Tried reducing emission probability to 0.05. (Explanation of hyperparameters are added to overleaf). Also tried modified batch_size, learning rate, number of neurons,
