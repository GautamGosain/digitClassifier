#MNIST
# DataLoader, Transformation
# Multilayer NeuralNet, activation function
# Loss and optimizer
# training loop
# model evaluation
# gpu support

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# import sys
# import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("runs/mnist2")

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
input_size = 784 #28*28
hidden_size = 100
num_classes = 10  #10 digits
num_epochs = 2     # number of epochs are less as more examples/ large batches
batch_size = 100
learning_rate = 0.01

#MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)          #root is for storing dataset   #training dataset   #download if nt present
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)   # list of batches. 600 batches - each containing 100 examples
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)      # shape of samples is 100,1,28,28  (100 batch size, 1 colour, 28*28 -size)  # shape of labels = 100 (100 training examples in 1batch, so 1 output corresponding to each example)

# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(samples[i][0], cmap='gray')
# # plt.show()
# img_grid = torchvision.utils.make_grid(samples)
# writer.add_image('mnistimages', img_grid)
# writer.close()
# sys.exit()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, X):
        out = self.l1(X)
        out = self.relu(out)
        out = self.l2(out)
        # no softmax here as crossEntropyLoss will apply it
        return out

model = NeuralNet(input_size, hidden_size, num_classes)

lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# writer.add_graph(model, samples.reshape(-1,28*28))
# writer.close()
# sys.exit()
#trianing loop
n_total_steps = len(train_loader)

n_training_samples = 0
# running_loss = 0.0
# running_correct = 0

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):             # 600 batches containing 100 examples each
        #100,1,28,28 --> 100, 784
        images = images.reshape(-1, 28*28).to(device)    # -1 automatically adjust shapes accordingly
        labels = labels.to(device)
        n_training_samples += labels.shape[0]

        #forward
        outputs = model(images)
        loss = lossFn(outputs, labels)

        #backward
        optimizer.zero_grad() #set gradient zero
        loss.backward()
        optimizer.step()     # optimize parameters

        # running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        # running_correct += (predicted == labels).sum().item()

        if i%100 == 0:
            print(f'epoch {epoch+1}, step{i+1}/{n_total_steps}, loss = {loss}')
            # writer.add_scalar('training loss', running_loss/100, epoch*n_total_steps+i)
            # writer.add_scalar('accuracy', running_correct / 100, epoch * n_total_steps + i)
            # running_loss = 0.0
            # running_correct = 0

print(n_training_samples/num_epochs)         # 600 batches having 100 training examples each. total 60000, total iterations = num_epochs*60000


#testing
# labels_grp = []
# preds = []

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 784).to(device)
        labels = labels.to(device)
        outputs = model(images)
        index, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

        # class_predictions = [F.softmax(outputs, dim=0) for output in outputs]
        # preds.append(class_predictions)
        # labels_grp.append(predictions)

    # preds = torch.cat([torch.stack(batch) for batch in preds])     #10000 * 10
    # labels_grp = torch.cat(labels_grp)    #10000 * 1
    acc =  n_correct/n_samples

    print(acc, n_samples, n_correct)

    # classes = range(10)

    # for i in classes:
    #     labels_i = labels_grp ==  i
    #     preds_i = preds[:, i]
    #     writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
    #     writer.close()


