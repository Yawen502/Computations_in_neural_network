# Modifications to code:

1. Change the name of model class
2. Change name of code to be consistent with our model in overleaf
3. Alter self.Sigmoid to self.sigmoid
4. Change nonlinearity of r_t from Sigmoid to ReLU
5. Replace exp() with softplus()
6. Change rand()in the init to empty() to reduce confusion
7. Alter the initialisations. Change initialisations of matrices to glorot initialisations, and** set b_v to zero, set b_z to log1/99
8. For Dale's law, set the components which changed sign to zero.
