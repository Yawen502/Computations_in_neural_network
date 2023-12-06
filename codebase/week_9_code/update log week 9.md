Week 8 meeting notes:

Get familiar and do some run with SSH server. (Done)

We can crop the images(after training for 25%, erase the hidden state), or just only feed 25%, 50% etc. data to the model. Check how the accuracies decay for different models. (working memory test) Let’s say we use constant A bRNN, matrix A bRNN and vanilla and simple GRU for this one.

Separate dimensions:

Dales’ law CBRNN and non-Dales’ law CBRNN

A_{ij} vs constant A 

1. Dales’ law CBRNN, constant A : most biologically plausbile
2. do all of them except for Dales’, non-constant A

Try PSTP and see whether it has a memory problem. 
- no memory problem, accuracy decreases compared to the other models.

Do a MNIST test in fixed random order (random permutation)

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
