## multi-scale vanilla RNN and constant z vanilla RNN
for multi-scale vanilla RNN, $z_t = \sigma (b_z)$

for constant z vanilla RNN, z= constant = 1, 0.5, 0.1 
![perf_plot](https://github.com/Yawen502/Computations_in_neural_network/assets/71087503/69e64226-a5ae-47f2-bb7d-f00d7f583858)

## pretraining Dale-CB and then introduce STP feature

A new variable self.A is added, and it is initialised as 0. As the STP feature is active, self.A increases by 0.2 each time until it reaches 1.

