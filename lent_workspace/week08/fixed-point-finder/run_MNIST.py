'''
examples/torch/run_FlipFlop.py
Written for Python 3.8.17 and Pytorch 2.0.1
@ Matt Golub, June 2023
Please direct correspondence to mgolub@cs.washington.edu
'''

import pdb
import sys
import numpy as np

from FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder

input_size = 8
sequence_length = 28*28//input_size
hidden_size = 48
num_layers = 1
num_classes = 10
batch_size = 40
num_epochs = 1
learning_rate = 0.01
stride_number = 4

'Data Preprocessing'
import torch
import math
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# get index of currently selected device
print(torch.cuda.current_device()) # returns 0 in my case

# get number of GPUs available
print(torch.cuda.device_count()) # returns 1 in my case

# get the name of the device
print(torch.cuda.get_device_name(0)) # good old Tesla K80

def snake_scan(img):
    'Converts a 32x32 image to a 32x96 array with snake-like row ordering'
    if len(img.shape) != 3:
        raise ValueError('Input image must be a 3D array')
    
    channels, rows, cols = img.shape
    snake = np.zeros((rows, cols * channels), dtype=img.dtype)
    for r in range(rows):
        row_data = img[:, r, :].flatten()  # Flattening each row into a 1D array of 96 elements
        if r % 2 == 1:
            row_data = row_data[::-1]  # Reversing the order for alternate rows
        snake[r] = row_data
    return snake

def stride(input_data, stride):
    'turn [batch_size, sequence_length, input_size] into [batch_size, sequence_length*input_size/stride, input_size]'
    batch_size, sequence_length, input_size = input_data.shape
    # flatten the input data to put sequence and input size together
    input_data = input_data.reshape(batch_size, -1)
    # append zeros to make sure the last pixel can be fed as the first pixel of the next sequence
    n = input_size - (sequence_length*input_size)%stride

    input_data = input_data.cpu()
    input_data = input_data.numpy()
    input_data = np.append(input_data, np.zeros((batch_size, n)), axis=1)
    input_data = torch.tensor(input_data)
    #print(input_data.shape)
    output_data = torch.zeros(batch_size, sequence_length*input_size//stride, input_size)
    for i in range(sequence_length*input_size//stride):
        # if stride = input size, then the output data is the same as input data
        #print(i)

        output_data[:,i,:] = input_data[:,i*stride:i*stride+input_size]
        #print(output_data[batch,i,:])

    return output_data

from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import Lambda

transform = transforms.Compose([
    ToTensor(),
    Lambda(lambda x: torch.tensor(snake_scan(x.numpy())))
])

train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    download = True,
    transform = transform     
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    download = True,
    transform = transform
)



'Hyperparameters'
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          num_workers=0),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=batch_size, 
                                          shuffle=False, 
                                          num_workers=0),
}
loaders
'''
for i, (images, labels) in enumerate(loaders['train']):
    images = images.reshape(-1, sequence_length, input_size).to(device)
    images = stride(images, stride_number).to(device)
    print(images.shape)
    print(labels.shape)
    print(len(loaders['train']))
    break
'''


class Dale_CBcell(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Dale_CBcell, self).__init__()
        self.hidden_size = hidden_size
    
        ### Parameters ###
        # voltage gate v_t 
        self.P = torch.nn.Parameter(torch.empty(self.hidden_size, input_size))           
        self.b_v = torch.nn.Parameter(torch.zeros(self.hidden_size, 1))   

        # Update gate z_t
        # K and W are unbounded free parameters   
        # C represents  current based portion of connectivity       
        self.K = torch.nn.Parameter(self.init_dale(self.hidden_size, self.hidden_size))
        self.C = torch.nn.Parameter(self.init_dale(self.hidden_size, self.hidden_size))
        self.P_z = torch.nn.Parameter(torch.empty(self.hidden_size, input_size))
        self.b_z = torch.nn.Parameter(torch.empty(self.hidden_size, 1))   
        # Potentials are initialised with right signs
        self.e_e = torch.nn.Parameter(torch.rand(1))
        self.e_i = torch.nn.Parameter(-torch.rand(1))

        # Firing rate, Scaling factor and time step initialization
        self.v_t = torch.zeros(1, self.hidden_size, dtype=torch.float32)

        # dt is a constant
        self.dt = nn.Parameter(torch.tensor(0.1), requires_grad = False)

        ### Nonlinear functions ###
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

        ### Initialisation ###
        glorot_init = lambda w: nn.init.uniform_(w, a=-(1/math.sqrt(hidden_size)), b=(1/math.sqrt(hidden_size)))
        positive_glorot_init = lambda w: nn.init.uniform_(w, a=0, b=(1/math.sqrt(hidden_size)))

        # initialise matrices
        # P and P_z are unconstrained
        for w in self.P_z, self.P:
            glorot_init(w)
        for w in self.K, self.C:
            positive_glorot_init(w)
        # init b_z to be log 1/99
        nn.init.constant_(self.b_z, torch.log(torch.tensor(1/99)))

        #self.v_t_history = []
        #self.z_t_history = []

    def init_dale(self, rows, cols):
        # Dale's law with equal excitatory and inhibitory neurons
        exci = torch.empty((rows, cols//2)).exponential_(1.0)
        inhi = -torch.empty((rows, cols//2)).exponential_(1.0)
        weights = torch.cat((exci, inhi), dim=1)
        weights = self.adjust_spectral(weights)
        return weights

    def adjust_spectral(self, weights, desired_radius=1.5):
        #values, _ = torch.linalg.eig(weights @ weights.T)
        values = torch.linalg.svdvals(weights)
        radius = values.abs().max()
        return weights * (desired_radius / radius)
        

    @property
    def r_t(self):
        return self.sigmoid(self.v_t)

    def forward(self, hidden, x):    
        print(hidden.shape)
        print(x.shape)
        self.v_t = hidden    
        if self.v_t.dim() == 3:           
            self.v_t = self.v_t[0]
        self.v_t = torch.transpose(self.v_t, 0, 1)

        ### Constraints###
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
        self.W = W

        ### Update Equations ###
        input_mask = torch.ones_like(self.P)
        input_mask[self.hidden_size//4:self.hidden_size//2,:] = 0
        input_mask[3*self.hidden_size//4:,:] = 0
        P = self.P * input_mask

        self.z_t = torch.zeros(self.hidden_size, 1)
        self.z_t = self.dt * self.sigmoid(torch.matmul(K , self.r_t) + torch.matmul(self.P_z, x) + self.b_z)
        self.v_t = (1 - self.z_t) * self.v_t + self.dt * (torch.matmul(W, self.r_t) + torch.matmul(P, x) + self.b_v)
        self.v_t = torch.transpose(self.v_t, 0, 1)      
        excitatory = self.v_t[:, :self.hidden_size//2]
        self.excitatory = torch.cat((excitatory, torch.zeros_like(excitatory)), 1)  
        return self.v_t

class Dale_CB_batch(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(Dale_CB_batch, self).__init__()
        self.rnncell = Dale_CBcell(input_size, hidden_size, num_layers).to(device)
        self.batch_first = batch_first

    def forward(self, batch_input):
        if self.batch_first == True:
            for n in range(batch_input.size(1)):
                #print(x.shape)
                x_slice = torch.transpose(batch_input[:,n,:], 0, 1)
                self.rnncell(x_slice)
        return self.rnncell.excitatory            
            
class Dale_CB(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Dale_CB, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = Dale_CB_batch(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, 10)
        self.batch_first = True
        pass

    def forward(self, x):
        # Set initial hidden and cell states 
        self.lstm.rnncell.v_t = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Passing in the input and hidden state into the model and  obtaining outputs
        out = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        #Reshaping the outputs such that it can be fit into the fully connected layer
        # output mask
        output_mask = torch.ones_like(out)
        output_mask[:,:self.hidden_size//4] = 0
        output_mask[:,3*self.hidden_size//4:] = 0        
        out = out * output_mask
        out = self.fc(out)
        return out.squeeze(-1)
        
        pass                                    
pass

model = Dale_CB(input_size, hidden_size, num_layers, num_classes).to(device)

'Training'
print(model)
loss_func = nn.CrossEntropyLoss()

# load parameters from pth file
model.load_state_dict(torch.load('05_Dale-CB_48.pth'))

fpf_hps = {
    'max_iters': 10000,
    'lr_init': 1.,
    'outlier_distance_scale': 10.0,
    'verbose': True, 
    'super_verbose': True}

fpf = FixedPointFinder(model, **fpf_hps)
images, labels = next(iter(loaders['train']))
images = images.reshape(-1, sequence_length, input_size).to(device)
print(images.shape)
images = images[-1].reshape(1, 98, 8)
fps = fpf.find_fixed_points(initial_states= torch.zeros(1, hidden_size).to(device), inputs = images)

fps.plot()