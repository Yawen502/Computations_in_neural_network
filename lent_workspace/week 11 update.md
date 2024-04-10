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

#### Results
After tuning the hyperparameter to make the task simpler, it is still too hard for all the networks to do.
For example, LSTM had this result after 1000 epochs when number of neuron = 100, batch size = 128, p_emit=0.05, t_window = 5, t_relax = 100.
![image](https://github.com/Yawen502/Computations_in_neural_network/assets/71087503/700d434e-a215-415b-ba95-d0cbecdae6ff)

MSE loss: 2.02e-01 (When this reaches 1e-04 the training is successful.)

Training trajectories:

                Epoch 995; loss: 2.02e-01; grad norm: 1.25e-01; learning rate: 9.63e-05; time: 3.31e-01s
                Epoch 996; loss: 2.02e-01; grad norm: 1.28e-01; learning rate: 9.63e-05; time: 2.21e-01s
                Epoch 997; loss: 2.02e-01; grad norm: 1.40e-01; learning rate: 9.15e-05; time: 2.40e-01s
                Epoch 998; loss: 2.02e-01; grad norm: 1.75e-01; learning rate: 9.15e-05; time: 2.41e-01s
                Epoch 999; loss: 2.02e-01; grad norm: 1.13e-01; learning rate: 9.15e-05; time: 2.31e-01s
                Epoch 1000; loss: 2.02e-01; grad norm: 1.31e-01; learning rate: 9.15e-05; time: 3.00e-01s
                Epoch 1001; loss: 2.01e-01; grad norm: 9.58e-02; learning rate: 9.15e-05; time: 2.31e-01s

### time gap in snake MNIST:
*use time gap of 24, 48, 96 — does not drag down the performance of multiscale RNN*

### Pushing up network performance:
increasing number of neuron to 100  push the network performance to

Multi-scale RNN: 86.28%

CB-RNN-tied: 80.40%

simple GRU: 87.02%

Conclusion: Increase of number of neuron can increase network performance greatly, but the number of epochs is still within 10 epochs when the performance starts to drop. 

Example 1 simple GRU:

    Epoch [1/20], Step [750/1500], Training Accuracy: 65.20
    Epoch [1/20], Step [1500/1500], Training Accuracy: 79.80
    Epoch [2/20], Step [750/1500], Training Accuracy: 84.80
    Epoch [2/20], Step [1500/1500], Training Accuracy: 86.00
    Epoch [3/20], Step [750/1500], Training Accuracy: 86.70
    Epoch [3/20], Step [1500/1500], Training Accuracy: 88.40
    Epoch [4/20], Step [750/1500], Training Accuracy: 88.90
    Epoch [4/20], Step [1500/1500], Training Accuracy: 89.40
    Epoch [5/20], Step [750/1500], Training Accuracy: 88.60
    Epoch [5/20], Step [1500/1500], Training Accuracy: 90.50
    Epoch [6/20], Step [750/1500], Training Accuracy: 90.90
    Epoch [6/20], Step [1500/1500], Training Accuracy: 88.60
    Epoch [7/20], Step [750/1500], Training Accuracy: 90.60
No improvement in validation accuracy for 2 epochs. Stopping training.

Example 2 CB-RNN-tied:

    Epoch [1/20], Step [750/1500], Training Accuracy: 58.40
    Epoch [1/20], Step [1500/1500], Training Accuracy: 69.00
    Epoch [2/20], Step [750/1500], Training Accuracy: 70.70
    Epoch [2/20], Step [1500/1500], Training Accuracy: 74.50
    Epoch [3/20], Step [750/1500], Training Accuracy: 74.50
    Epoch [3/20], Step [1500/1500], Training Accuracy: 79.20
    Epoch [4/20], Step [750/1500], Training Accuracy: 80.40
    Epoch [4/20], Step [1500/1500], Training Accuracy: 78.80
    Epoch [5/20], Step [750/1500], Training Accuracy: 46.30
    No improvement in validation accuracy for 2 epochs. Stopping training.



### Implementing behavioural flexibility task
Not done - do it next week

