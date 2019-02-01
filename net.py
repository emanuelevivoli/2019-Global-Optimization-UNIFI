import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


class Net(nn.Module):

    def __init__(self, max_epochs, learning_rate, useGPU=True):
        super(Net, self).__init__()

        # net layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # loss function
        self.criterion = nn.CrossEntropyLoss()
        # loss optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # max training epochs
        self.max_epochs = max_epochs

        # selection of device to use
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.useGPU = useGPU
        if self.device == "cpu":
            self.useGPU = False

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def fit(self, trainloader):
        losses = []
        for epoch in range(self.max_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):

                # get the inputs
                inputs, labels = data

                if self.useGPU:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # loss update
                running_loss += loss.item()

                # print statistics
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

            losses.append(running_loss)

        return losses

    def eval_metrics(self, testloader):

        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for data in testloader:

                # get some test images
                images, labels = data
                if self.useGPU:
                    images, labels = images.to(self.device), labels.to(self.device)

                # images classes prediction
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)

                # loss update
                loss += self.criterion(outputs, labels).item()

                # update numbers of total and correct predictions
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return round(accuracy, 1), loss

