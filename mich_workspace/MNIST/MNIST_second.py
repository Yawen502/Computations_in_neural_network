import torch
import math
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# get index of currently selected device
print(torch.cuda.current_device()) # returns 0 in my case

# get number of GPUs available
print(torch.cuda.device_count()) # returns 1 in my case

# get the name of the device
print(torch.cuda.get_device_name(0)) # good old Tesla K80

from torchvision import datasets
from torchvision.transforms import ToTensor
train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)


from torch.utils.data import DataLoader
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=0),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=0),
}
loaders

from torch import nn
import torch.nn.functional as F

sequence_length = 28
input_size = 28
hidden_size = 48
num_layers = 1
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.005


class customGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(customGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.r_t = torch.zeros(1, self.hidden_size, dtype=torch.float32)
        # Rest gate r_t 
        self.w_r = torch.nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.p_r = torch.nn.Parameter(torch.rand(self.hidden_size, input_size))           
        self.b_r = torch.nn.Parameter(torch.rand(self.hidden_size, 1))   

        # Update gate z_t
        # Wz is defined in the forward function          
        self.g_z = torch.nn.Parameter(torch.rand(self.hidden_size, 1))           

        # Firing rate, Scaling factor and time step initialization
        self.dt = nn.Parameter(torch.clamp(torch.tensor(0.01), min = 0.0, max = 1.0), requires_grad= True)

        # Nonlinear functions
        self.Sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()
        for name, param in self.named_parameters():
            nn.init.uniform_(param, a=-(1/math.sqrt(hidden_size)), b=(1/math.sqrt(hidden_size)))

    def forward(self, x):        
        if self.r_t.dim() == 3:           
            self.r_t = self.r_t[0]
        self.r_t = torch.transpose(self.r_t, 0, 1)
        self.dt = nn.Parameter(torch.clamp(self.dt, min=0.0, max=1.0))

        self.z_t = self.dt * self.Sigmoid(self.g_z)
        self.r_t = (1 - self.z_t) * self.r_t + self.z_t * self.Tanh(torch.matmul(self.w_r, self.r_t) + torch.matmul(self.p_r, x) + self.b_r)
        self.r_t = torch.transpose(self.r_t, 0, 1)                

class customGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(customGRU, self).__init__()
        self.rnncell = customGRUCell(input_size, hidden_size, num_layers).to(device)
        self.batch_first = batch_first

    def forward(self, x):
        if self.batch_first == True:
            for n in range(x.size(1)):
                x_slice = torch.transpose(x[:,n,:], 0, 1)
                self.rnncell(x_slice)
        return self.rnncell.r_t             
            
class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = customGRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)
        pass

    def forward(self, x):
        # Set initial hidden and cell states 
        self.lstm.rnncell.r_t = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Passing in the input and hidden state into the model and  obtaining outputs
        out = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        #Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out)
        return out
        
        pass                                    
pass
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
print(model)
loss_func = nn.CrossEntropyLoss()

from torch import optim
optimizer = optim.Adam(model.parameters(), lr = 0.01)   

def train(num_epochs, model, loaders):
        
    # Train the model
    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = loss_func(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                pass
        
        pass
    pass
train(num_epochs, model, loaders)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in loaders['test']:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total = total + labels.size(0)
        correct = correct + (predicted == labels).sum().item()
print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


"""
for name, param in model.named_parameters():
    print(name, param)
torch.save(model.state_dict(), 'model_parameters.pth')
wz = torch.abs(model.state_dict()['lstm.rnncell.w_r'])
print(wz)
A = torch.exp(-model.state_dict()['lstm.rnncell.a'])
print(A)
"""
# Test Accuracy of the model on the 10000 test images: 92.93 %