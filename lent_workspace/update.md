1. separate time window and time relaxation and see which one cause more problem.
2. Keep generating new training dataset and validation set, do not use repeated data, and use a stopping criterion. (either MSE loss is small enough, or within (relative error = MSE/variance(target))
3. save parameters and continue again if necessary (since the effective epochs are small, this is not necessary at this stage)

## LSTM
Hyperparameters:

                 n_bits=2,
                 n_time=100,
                 p=0.5,
                 random_seed=0,
                 t_window = 10,
                 t_relax = 50


![image](https://github.com/Yawen502/Computations_in_neural_network/assets/71087503/16af74bf-9f02-454c-991d-cfb600807095)

![image](https://github.com/Yawen502/Computations_in_neural_network/assets/71087503/b748821f-717c-4f51-adc8-92411a597309)

                 n_bits=2,
                 n_time=100,
                 p=0.5,
                 random_seed=0,
                 t_window = 10,
                 t_relax = 30
                 
![image](https://github.com/Yawen502/Computations_in_neural_network/assets/71087503/a58b6cbf-6474-4022-a2f2-e24e41948b47)

![image](https://github.com/Yawen502/Computations_in_neural_network/assets/71087503/6e5d904a-83da-47ea-87d3-94a1dd8d81df)

## Vanilla

                 n_bits=2,
                 n_time=100,
                 p=0.5,
                 random_seed=0,
                 t_window = 10,
                 t_relax = 50
                 

![image](https://github.com/Yawen502/Computations_in_neural_network/assets/71087503/c0b43a09-f024-4cd3-9f8f-5e6aae97aa27)

![image](https://github.com/Yawen502/Computations_in_neural_network/assets/71087503/52d5dd94-9e62-48ab-9700-06817be4e439)


## Multiscale RNN

                 n_bits=2,
                 n_time=100,
                 p=0.5,
                 random_seed=0,
                 t_window = 10,
                 t_relax = 50

![image](https://github.com/Yawen502/Computations_in_neural_network/assets/71087503/ce8423e8-2417-4918-9daf-c4a4acd630fc)
