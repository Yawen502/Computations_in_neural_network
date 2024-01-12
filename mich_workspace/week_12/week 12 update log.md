# Modifications:

1. Change the initialisation method of Dale's law.
   
               def init_dale(self, rows, cols):
                    # Dale's law with equal excitatory and inhibitory neurons
                    exci = torch.empty((rows, cols//2)).exponential_(1.0)
                    inhi = -torch.empty((rows, cols//2)).exponential_(1.0)
                    weights = torch.cat((exci, inhi), dim=1)
                    weights = self.adjust_spectral(weights)
                    return weights
            
                def adjust_spectral(self, weights, desired_radius=1.5):
                    values, _ = torch.linalg.eig(weights @ weights.T)
                    radius = values.abs().max()
                    return weights * (desired_radius / radius)

2.  Add lines at the end to retrieve the parameters:
   
               # Retrieve weights
               P = model.lstm.rnncell.P.detach().cpu().numpy()
               W = model.lstm.rnncell.W.detach().cpu().numpy()
               read_out = model.fc.weight.detach().cpu().numpy()
               
               torch.save({
                   'Weight Matrix W': W,
                   'Input Weight Matrix P': P,
                   'Readout Weights': read_out,
               },'Dale-CB-weights.pth')
# Results
<img width="644" alt="image" src="https://github.com/Yawen502/Computations_in_neural_network/assets/71087503/e5ed573c-1b96-41f1-bb74-49bf425f025a">

sigmoid_Dale-CB-STP:

<img width="500" alt="image" src="https://github.com/Yawen502/Computations_in_neural_network/assets/71087503/028d1c7f-716f-44c2-aeb3-acbd1cd85071">

sigmoid_CB-RNN-tied:

<img width="500" alt="image" src="https://github.com/Yawen502/Computations_in_neural_network/assets/71087503/95d8fae6-013a-46d7-9400-b63b9035902c">

relu_CB-RNN-tied:

<img width="500" alt="image" src="https://github.com/Yawen502/Computations_in_neural_network/assets/71087503/602ae6d9-ff78-48fd-baaa-9d71cd304e61">
