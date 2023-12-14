# The first model gives the conductance model as we discussed
# The second model is an simplified one using constant g
# The third model is the simple GRU model, should be the strongest one

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

input_size = 16
sequence_length = 28*28//input_size
hidden_size = 24
num_layers = 1
num_classes = 10
batch_size = 40
num_epochs = 10
learning_rate = 0.01
stride_number = 4

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



'Model Definition'
class customGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, complexity, num_layers):
        super(customGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.complexity = complexity
        self.ones = torch.ones(self.hidden_size, self.hidden_size)
        self.batch_size = batch_size 
        self.forprintingX = []
        self.forprintingU = []
        self.forprintingh = []
        # Nonlinear functions
        self.Sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()
        self.relu = nn.ReLU()


        # STP model initialisations
        if self.complexity == "rich":
            # Short term Plasticity variables 
            self.delta_t = 1
            self.z_min = 0.001
            self.z_max = 0.1

            # Short term Depression parameters  
            self.c_x = torch.nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))

            # Short term Facilitation parameters
            self.c_u = torch.nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
            self.c_U = torch.nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
            
            # State initialisations
            self.X = torch.ones(self.hidden_size, self.hidden_size, dtype=torch.float32)     
            self.U = torch.full((self.hidden_size, self.hidden_size), 0.9, dtype=torch.float32)         
            self.Ucap = 0.9 * self.Sigmoid(self.c_U)
            self.Ucapclone = self.Ucap.clone().detach()

        if self.complexity == "poor":
            # Short term Plasticity variables 
            self.delta_t = 1
            self.z_min = 0.001
            self.z_max = 0.1

            # Short term Depression parameters  
            self.c_x = torch.nn.Parameter(torch.rand(self.hidden_size, 1))

            # Short term Facilitation parameters
            self.c_u = torch.nn.Parameter(torch.rand(self.hidden_size, 1))
            self.c_U = torch.nn.Parameter(torch.rand(self.hidden_size, 1))
            
            # State initialisations
            self.X = torch.ones(self.hidden_size, 1, dtype=torch.float32)
            self.U = torch.full((self.hidden_size, 1), 0.9, dtype=torch.float32)   
            self.Ucap = 0.9 * self.Sigmoid(self.c_U)
            self.Ucapclone = self.Ucap.clone().detach()
            
        self.w_r = torch.nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.p_r = torch.nn.Parameter(torch.rand(self.hidden_size, input_size))           
        self.b_r = torch.nn.Parameter(torch.full((self.hidden_size, 1), -torch.log(torch.tensor(99.0))), requires_grad=True)

        # Initialize b_r to be -log(99), so at steady state f.r. close to biological firing rate of 1Hz

        # Update gate z_t
        # K is always positive            
        self.g_z = torch.nn.Parameter(torch.rand(self.hidden_size, 1))     
        self.K = torch.nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))   
        self.p_z = torch.nn.Parameter(torch.rand(self.hidden_size, input_size))

        # Firing rate, Scaling factor and time step initialization
        self.v_t = torch.zeros(1, self.hidden_size, dtype=torch.float32)
        # a and dt are fixed
        
        self.dt = torch.nn.Parameter(torch.tensor(0.1), requires_grad = False)
        # dt is clamped between 0 and 1 to ensure it makes sense biologically

        self.w_scale = 0.192
        self.b = 10/ self.w_scale

        self.zt_history = []
        self.ht_history = []
        for name, param in self.named_parameters():
            nn.init.uniform_(param, a=-(1/math.sqrt(hidden_size)), b=(1/math.sqrt(hidden_size)))
    @property
    def r_t(self):
        return self.Sigmoid(self.v_t)

    def forward(self, x):        
        if self.v_t.dim() == 3:           
            self.v_t = self.v_t[0]

        self.v_t = torch.transpose(self.v_t, 0, 1)
        # Apply constraints to follow Dale's principle
        # w_r and k are indepedent and always positive
        # determine A based on the sign of w_r 
                    
        # STP model updates
        if self.complexity == "rich":
            # Short term Depression 
            self.z_x = self.z_min + (self.z_max - self.z_min) * self.Sigmoid(self.c_x)
            self.X = self.z_x + torch.mul((1 - self.z_x), self.X) - self.delta_t * self.U * torch.einsum("ijk, ji  -> ijk", self.X, self.r_t)

            # Short term Facilitation 
            self.z_u = self.z_min + (self.z_max - self.z_min) * self.Sigmoid(self.c_u)    
            self.Ucap = 0.9 * self.Sigmoid(self.c_U)
            self.U = self.Ucap * self.z_u + torch.mul((1 - self.z_u), self.U) + self.delta_t * self.Ucap * torch.einsum("ijk, ji  -> ijk", (1 - self.U), self.r_t)
            self.Ucapclone = self.Ucap.clone().detach() 
            self.U = torch.clamp(self.U, min=self.Ucapclone.repeat(self.U.size(0), 1, 1).to(device), max=torch.ones_like(self.Ucapclone.repeat(self.U.size(0), 1, 1).to(device)))
            

            # Update gate z_t
            self.z_t = self.dt * self.Sigmoid(torch.matmul(self.K, self.r_t) + torch.matmul(self.p_z, x) + self.g_z)
            # Voltage update after both conductance and STP updates
            self.v_t = (1 - self.z_t) * self.v_t + self.dt * (torch.matmul(self.X*self.U*self.w_r, self.r_t) + torch.matmul(self.p_r, x) + self.b_r)
            self.v_t = self.v_t[0]
            self.v_t = torch.transpose(self.v_t, 0, 1) 


        if self.complexity == "poor":
            x = torch.transpose(x, 0, 1)
            sigmoid = nn.Sigmoid()
            
            # Short term Depression 
            self.z_x = self.z_min + (self.z_max - self.z_min) * sigmoid(self.c_x)
            self.X = self.z_x + torch.mul((1 - self.z_x), self.X) - self.delta_t * self.U * self.X * self.r_t

            # Short term Facilitation 
            self.z_u = self.z_min + (self.z_max - self.z_min) * sigmoid(self.c_u)    
            self.Ucap = 0.9 * sigmoid(self.c_U)
            self.U = self.Ucap * self.z_u + torch.mul((1 - self.z_u), self.U) + self.delta_t * self.Ucap * (1 - self.U) * self.r_t
            self.Ucapclone = self.Ucap.clone().detach()
            self.U = torch.clamp(self.U, min=self.Ucapclone.repeat(1, x.size(0)).to(device), max=torch.ones_like(self.Ucapclone.repeat(1, x.size(0)).to(device)))
            x = torch.transpose(x, 0, 1)
            # Update gate z_t
            self.z_t = self.dt * sigmoid(torch.matmul(self.K, self.r_t) + torch.matmul(self.p_z, x) + self.g_z)
            # Voltage update after both conductance and STP updates
            self.v_t = (1 - self.z_t) * self.v_t + self.dt * (torch.matmul(self.w_r, self.U*self.X*self.r_t) + torch.matmul(self.p_r, x) + self.b_r)
            self.v_t = torch.transpose(self.v_t, 0, 1) 


                


class customGRU(nn.Module):
    def __init__(self, input_size, hidden_size, complexity, num_layers, batch_first=True):
        super(customGRU, self).__init__()
        self.rnncell = customGRUCell(input_size, hidden_size, complexity, num_layers).to(device)
        self.batch_first = batch_first

    def forward(self, x):
        if self.batch_first == True:
            for n in range(x.size(1)):
                #print(x.shape)
                x_slice = torch.transpose(x[:,n,:], 0, 1)
                self.rnncell(x_slice)
        return self.rnncell.v_t             
            
class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, complexity, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = customGRU(input_size, hidden_size, complexity, num_layers)
        self.fc = nn.Linear(hidden_size, 10)
        pass

    def forward(self, x):
        # Set initial hidden and cell states 
        if self.lstm.rnncell.complexity == "rich":
            self.lstm.rnncell.X = torch.ones(x.size(0), self.hidden_size, self.hidden_size, dtype=torch.float32).to(device)
            self.lstm.rnncell.U = (self.lstm.rnncell.Ucapclone.repeat(x.size(0), 1, 1)).to(device)
            self.lstm.rnncell.v_t = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        if self.lstm.rnncell.complexity == "poor":
            self.lstm.rnncell.X = torch.ones(self.hidden_size, x.size(0), dtype=torch.float32).to(device)
            self.lstm.rnncell.U = (self.lstm.rnncell.Ucapclone.repeat(1, x.size(0))).to(device)
            self.lstm.rnncell.v_t = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
            # Passing in the input and hidden state into the model and  obtaining outputs
        out = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        #Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out)
        return out.squeeze(-1)
        
        pass                                    
pass
model = RNN(input_size, hidden_size, 'poor', num_layers, num_classes).to(device)
print(model)
loss_func = nn.CrossEntropyLoss()

'Trajactory Tracking and Training'
from torch import optim
model_optimizer = optim.Adam(model.parameters(), lr = learning_rate)   

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

def subset_loader(full_dataset, batch_size, subset_ratio=0.1):
    # Generate labels array to use in stratified split
    labels = []
    for _, label in full_dataset:
        labels.append(label)
    labels = np.array(labels)

    # Perform a stratified shuffle split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=subset_ratio, random_state=0)
    for train_index, test_index in sss.split(np.zeros(len(labels)), labels):
        stratified_subset_indices = test_index

    # Create a Subset instance with the stratified subset indices
    stratified_subset = Subset(full_dataset, stratified_subset_indices)

    # Create DataLoader from the subset
    subset_loader = DataLoader(
        stratified_subset,
        batch_size=batch_size,
        shuffle=False  # No need to shuffle as we already have a random subset
    )

    return subset_loader
subtest = subset_loader(test_data, batch_size)

def evaluate_while_training(model, loaders):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in subtest:
            images = images.reshape(-1, sequence_length, input_size).to(device)
            images = stride(images, stride_number).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def train(num_epochs, model, loaders, patience=2, min_delta=0.01):
    model.train()
    total_step = len(loaders['train'])
    train_acc = []
    best_acc = 0
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            images = images.reshape(-1, sequence_length, input_size).to(device)
            images = stride(images, stride_number).to(device)
            labels = labels.to(device)
            model.train()
            # Forward pass
            outputs = model(images)
            loss = loss_func(outputs, labels)

            # Backward and optimize
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
            
            if (i+1) % 750 == 0:
                accuracy = evaluate_while_training(model, loaders)
                train_acc.append(accuracy)
                print('Epoch [{}/{}], Step [{}/{}], Training Accuracy: {:.2f}' 
                      .format(epoch + 1, num_epochs, i + 1, total_step, accuracy))

                # Check for improvement
                if accuracy - best_acc > min_delta:
                    best_acc = accuracy
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                if no_improve_epochs >= patience:
                    print("No improvement in validation accuracy for {} epochs. Stopping training.".format(patience))
                    return train_acc

    return train_acc

train_acc = train(num_epochs, model, loaders)

'Testing Accuracy'
# Test the model
model.eval()
with torch.no_grad():
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loaders['test']:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        images = stride(images, stride_number).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted ==labels).sum().item()

test_acc = 100 * correct / total
print('Accuracy of the model:{}%'.format(test_acc))
with open('result.csv', 'a') as f:
    f.write('01_simple_GRU with input size:{}, test accuracy:{}\n'.format(input_size, test_acc))




"""
Epoch [10/10], Step [400/500], Training Accuracy: 21.40
Epoch [10/10], Step [500/500], Training Accuracy: 22.50
Accuracy of the model:23.43%
Epoch [10/10], Step [400/500], Training Accuracy: 50.20
Epoch [10/10], Step [500/500], Training Accuracy: 48.40
Accuracy of the model:49.29%
"""

# stride length 4
# input length 4, Accuracy of the model:89.72%
# input length 8, Accuracy of the model:89.59%
# input length 12, Accuracy of the model:91.71% 

# Check if W and K and C are as we expect
