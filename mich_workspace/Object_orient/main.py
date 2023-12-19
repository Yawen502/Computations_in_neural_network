from packages.dataset_preprocessing import DatasetPreprocessor
from packages.simple_GRU import *
from packages.train import *
from packages.test import *
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

# Load data
preprocessor = DatasetPreprocessor()
loaders = preprocessor.load_data()

# Hyper-parameters
sequence_length = 32
input_size = 3
hidden_size = 128
num_layers = 1
num_classes = 10
batch_size = 100
learning_rate = 0.001
epochs = 10

# Model definition
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
print(model)

# Training
train = Train_and_track(model, learning_rate, batch_size, sequence_length, input_size)
train.train(epochs, loaders)
# Testing
tester = Tester(model, loaders, device, sequence_length, input_size)
tester.test_model()