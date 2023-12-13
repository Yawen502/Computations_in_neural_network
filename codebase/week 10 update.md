# Week 10 update log and Summary

Do a comparison graph:
-simple GRU
-full GRU
-vanilla RNN
-STP
-DaleCB(2 cases) CA/MA
-CB(2 cases) CA/MA
-STPDaleCB, STPCB
-SSTP (we're not going to do it)

Think of a good presentation form of the graph, try to make it not too busy. Do some summary plots in other forms e.g. bar plot

## Value updates
reduce batch size to 40 (lower batch size takes too long to train)
reduce validation to 2 times per epoch or so(read through the website)
working memory: examine conductance variations over time. We can examine either parameters or dynamic variables
keep input size below half the row length(thats the largest)
implement stride and use stride less than half of the input length
we can use 2 conventions of stride, 1. **always 4 (for cifar that is 12)** 2. **half of the input length**
stride = 4, with input length (4, 8 and 16)
decide between permuted MNIST and CIFAR( which is easier)
