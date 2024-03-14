## multi-scale vanilla RNN and constant z vanilla RNN
for multi-scale vanilla RNN, $z_t = \sigma (b_z)$

for constant z vanilla RNN, z= constant = 1, 0.5, 0.1 
![perf_plot](https://github.com/Yawen502/Computations_in_neural_network/assets/71087503/69e64226-a5ae-47f2-bb7d-f00d7f583858)

## pretraining Dale-CB and then introduce STP feature

A new variable self.A is added, and it is initialised as 0. As the STP feature is active, self.A increases by 0.2 each time until it reaches 1.

Comparison of model with/without pretraining and STP feature:
![perf_pretraining](https://github.com/Yawen502/Computations_in_neural_network/assets/71087503/eecc8fd3-8c56-42b5-b9f0-7e1e3b04f153)
![heatmap_W](https://github.com/Yawen502/Computations_in_neural_network/assets/71087503/8c5949de-8baf-4ff9-b1ab-a419a94737ce)


## FlipFlop Task with integration
The output side detects two consecutive pulses happening within t_window, and produces a pulse with the same sign. After t_relax, the pulse is reset to 0. Any later output pulse overwrites the previous one.

