# Week 10 update log and Summary

Do a comparison graph:
- simple GRU /
- full GRU /
- vanilla RNN /
- STP nonDale/Dale (MA)
- DaleCB MA only
- CB, CB-GRU refer to overleaf equation (40) /
- STPDaleCB MA

Think of a good presentation form of the graph, try to make it not too busy. Do some summary plots in other forms e.g. bar plot

## Value updates
reduce batch size to 40 (lower batch size takes too long to train)

reduce validation to 2 times per epoch or so(read through the website)

working memory: examine conductance variations over time. We can examine either parameters or dynamic variables

keep input size below half the row length(thats the largest)

implement stride and use stride less than half of the input length

we can use 2 conventions of stride, 1. **always 4 (for cifar that is 12)** 2. **half of the input length**
stride = 4, with input length (4, 8 and 16)
decide between permuted MNIST and CIFAR( which is easier): CIFAR

## Model Updates:
We have discussed in last meeting to replace W and A with K and the reversal potentials. Essentially we no longer have the conductance model, but the conductance-based AND current-based model.

### STP nonDale
STP is carried from David's code. The major idea about STP is that it's scaling the weight matrix $W$ in our previous model.
In this non Dale model, we want to keep $K$ to be positive to ensure the conductance is positive, but W and K are 
The current and conductance based part is represented as follow:

In initialisation, we now have

        self.K = torch.nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))  

And the parts about A are removed. In the forward loop we have

        K = torch.exp(self.K)
        ...
        self.z_t = self.dt * sigmoid(torch.matmul(K, self.r_t) + torch.matmul(self.p_z, x) + self.g_z)
        self.v_t = (1 - self.z_t) * self.v_t + self.dt * (torch.matmul(self.w_r, self.U*self.X*self.r_t) + torch.matmul(self.p_r, x) + self.b_r)

As the model does not follow Dale's law, w_r can take any arbitary values and signs. Note we choose p_z to be positive by now, but this might change in the future due to the introduction to the current-based model. (For conductance-based model the conductance always increases in response to x, so there's no doubt p_z should be positive. We should also consider whether p_z can decouple with p_r.) Here I take p_z independent of p_r because I think there's a considerable degree of freedom between these two. (?)


### DaleCB

After constructing arbitary matrices K and C as follows:\

        # Wz is defined in the forward function
        self.g_z = torch.nn.Parameter(torch.rand(self.hidden_size, 1))    
        self.p_z = torch.nn.Parameter(torch.rand(self.hidden_size, input_size))
        self.K = torch.nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.C = torch.nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        # v_e and v_i are constant
        self.v_e = torch.nn.Parameter(torch.tensor(1.0))
        self.v_i = torch.nn.Parameter(torch.tensor(1.0))
        

We then construct $W$ in the forward loop:

        K= torch.exp(self.K)
        C= torch.exp(self.C)
        v_e = torch.exp(self.v_e)
        v_i = -torch.exp(self.v_i)
        W_E = v_e * (K[:, :self.hidden_size//2] + C[:, :self.hidden_size//2])
        W_I = v_i * (K[:, self.hidden_size//2:] + C[:, self.hidden_size//2:])
        self.W = torch.cat((W_E, W_I), dim=1)

Other than this, the model stays the same as the STP Dale model.


### STPDale

This is simply combining the STP features and the Dale features.

### CB-GRU

We let $K$ and $W$ to be completely independent and dynamically updated. This model is very similar to simple GRU, and it aims to compare the difference between the voltage equations we have in our model, and the firing rate equations the simple GRU have. To make this comparison fairer, I also removed the constraints that p_z have to be positive.

            @property
            def r_t(self):
                return self.Sigmoid(self.v_t)
        ...
        self.z_t = self.dt * self.Sigmoid(torch.matmul(self.K , self.r_t) + torch.matmul(self.p_z, x) + self.g_z)
        self.v_t = (1 - self.z_t) * self.v_t + self.dt * (torch.matmul(self.w_r, self.r_t) + torch.matmul(self.p_r, x) + self.b_r)

Compare this with simple GRU:

        self.z_t = self.Sigmoid(torch.matmul(self.w_z, self.r_t) + torch.matmul(self.p_z, x) + self.b_z)
        self.r_t = (1 - self.z_t) * self.r_t + self.z_t * self.Sigmoid(torch.matmul(self.w_r, self.r_t) + torch.matmul(self.p_r, x) + self.b_r)
