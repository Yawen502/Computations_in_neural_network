import torch
from torch import optim
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

class Train_and_track:
    def __init__(self, model, learning_rate, batch_size, sequence_length, input_size, subset_ratio=0.1):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.subset_ratio = subset_ratio
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_func = nn.CrossEntropyLoss()
        self.sequence_length = sequence_length
        self.input_size = input_size

    def subset_loader(self, full_dataset):
        labels = []
        for _, label in full_dataset:
            labels.append(label)
        labels = np.array(labels)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.subset_ratio, random_state=0)
        for train_index, test_index in sss.split(np.zeros(len(labels)), labels):
            stratified_subset_indices = test_index

        stratified_subset = Subset(full_dataset, stratified_subset_indices)

        subset_loader = DataLoader(
            stratified_subset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return subset_loader

    def evaluate_while_training(self, loaders):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loaders['test']:
                images = images.reshape(-1, self.sequence_length, self.input_size).to(device)
                labels = labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

    def train(self, num_epochs, loaders, patience=5, min_delta=0.01):
        self.model.train()
        total_step = len(loaders['train'])
        train_acc = []
        best_acc = 0
        no_improve_epochs = 0

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(loaders['train']):
                images = images.reshape(-1, self.sequence_length, self.input_size).to(device)
                labels = labels.to(device)
                self.model.train()
                outputs = self.model(images)
                loss = self.loss_func(outputs, labels)

                self.model_optimizer.zero_grad()
                loss.backward()
                self.model_optimizer.step()

                if (i+1) % 100 == 0:
                    accuracy = self.evaluate_while_training(loaders)
                    train_acc.append(accuracy)
                    print('Epoch [{}/{}], Step [{}/{}], Training Accuracy: {:.2f}' 
                          .format(epoch + 1, num_epochs, i + 1, total_step, accuracy))

                    if accuracy - best_acc > min_delta:
                        best_acc = accuracy
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1

                    if no_improve_epochs >= patience:
                        print("No improvement in validation accuracy for {} epochs. Stopping training.".format(patience))
                        return train_acc

        return train_acc


