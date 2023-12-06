## Updates

### Get familiar and do some trials with SSH server.

### working memory best
We can crop the images(after training for 25%, erase the hidden state). Check how the accuracies decay for different models. (working memory test) 
Let’s say we use constant A bRNN, matrix A bRNN and vanilla and simple GRU for this one.![image](https://github.com/Yawen502/Computations_in_neural_network/assets/71087503/7aab119b-137c-4974-9454-388a111b9382)

### Try PSTP and see whether it has a memory problem. 
- no memory problem, accuracy decreases compared to the other models.

  
## What to do next
3 types of CBRNN:

Dales’ law CBRNN vs non-Dales’ law CBRNN, A_{ij} vs constant A 
Combining these two features gives four different types of CBRNN.

Dales’ law CBRNN, constant A is the most biologically plausbile
We do not consider the combination of Dales' law and non-constant A (literature not finished)


### Do a MNIST test in fixed random order (random permutation)

Do a comparison graph:

- simple GRU
- full GRU
- vanilla RNN
- STP model
- CBRNN(3 types)
- PSTP

Think of a good presentation form of the graph, try to make it not too busy.

Make a collection of all, then also

comparison between CBRNNs

take the best-performing ones

take the most biological ones

Do some summary plots in other forms e.g. bar plot
