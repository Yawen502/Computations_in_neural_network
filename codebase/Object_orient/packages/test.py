'Testing Accuracy'

class Tester:
    def __init__(self, model, loaders, device, sequence_length, input_size):
        self.model = model
        self.loaders = loaders
        self.device = device
        self.sequence_length = sequence_length
        self.input_size = input_size

    def test_model(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            correct = 0
            total = 0
            for images, labels in self.loaders['test']:
                images = images.reshape(-1, self.sequence_length, self.input_size).to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the model: {}%'.format(100 * correct / total))
        
