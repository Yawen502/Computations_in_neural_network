# Modifications to code:

1. Change the name of model class
2. Change name of code to be consistent with our model in overleaf
3. Alter self.Sigmoid to self.sigmoid
4. Change nonlinearity of r_t from Sigmoid to ReLU
   For models with ReLU, when combining it with STP, the model collapsed.
6. Replace exp() with softplus()
7. Change rand()in the init to empty() to reduce confusion
8. Alter the initialisations. Change initialisations of matrices to glorot initialisations, and** set b_v to zero, set b_z to log1/99

            ### Initialisation ###
            glorot_init = lambda w: nn.init.uniform_(w, a=-(1/math.sqrt(hidden_size)), b=(1/math.sqrt(hidden_size)))
            positive_glorot_init = lambda w: nn.init.uniform_(w, a=0, b=(1/math.sqrt(hidden_size)))
    
            # initialise matrices
            for w in self.W, self.P:
                glorot_init(w)
            for w in self.K, self.P_z:
                positive_glorot_init(w)
            # init b_z to be log 1/99
            nn.init.constant_(self.b_z, torch.log(torch.tensor(1/99)))
9. For Dale's law, set the components which changed sign to zero.

            ### Constraints ###
            K = self.softplus(self.K)
            C = self.softplus(self.C)
            # W is constructed using e*(K+C)
            W_E = self.e_e * (K[:, :self.hidden_size//2] + C[:, :self.hidden_size//2])
            W_I = self.e_i * (K[:, self.hidden_size//2:] + C[:, self.hidden_size//2:])
            # print to see which device the tensor is on
            # If sign of W do not obey Dale's law, then these terms to be 0
            W_E = self.relu(W_E)
            W_I = -self.relu(-W_I)
            W = torch.cat((W_E, W_I), 1)
