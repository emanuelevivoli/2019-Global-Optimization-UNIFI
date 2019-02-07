import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from tensorboardX import SummaryWriter


class Net(nn.Module):

    def __init__(self, max_epochs, learning_rate, weight_decay, useGPU=True):
        super(Net, self).__init__()

        # net layers
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # loss function
        self.criterion = nn.CrossEntropyLoss()
        # loss optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # max training epochs
        self.max_epochs = max_epochs

        self.tensorboard = SummaryWriter()

        # selection of device to use
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.useGPU = useGPU
        if self.device == "cpu":
            self.useGPU = False

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # patience: number of epochs without validation loss improvements for early stopping
    def fit(self, trainloader, validationloader, patience=-1):
        training_losses = []
        validation_losses = []
        best_validation_loss = 9999999999
        waited_epochs = 0
        for epoch in range(self.max_epochs):  # loop over the dataset multiple times

            ''' calculate training loss and do optimizer step '''
            training_loss = 0.0
            training_loss_updates = 0
            for i, data in enumerate(trainloader, 0):

                # get the inputs from training set
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

                # training loss update
                training_loss += loss.item()
                training_loss_updates += 1

            training_loss /= training_loss_updates
            training_losses.append(training_loss)

            ''' calculate validation loss '''
            validation_loss = 0.0
            validation_loss_updates = 0
            for i, data in enumerate(validationloader, 0):

                # get the inputs from validation set
                inputs, labels = data
                if self.useGPU:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                # predict batch labels
                outputs = self(inputs)

                # calculate batch loss
                loss = self.criterion(outputs, labels)

                # validation loss update
                validation_loss += loss.item()
                validation_loss_updates += 1
            validation_loss /= validation_loss_updates
            validation_losses.append(validation_loss)

            ''' print and save info of this epoch '''
            print("epoch " + str(epoch+1) + "/" + str(self.max_epochs) + ": training_loss=" + str(training_loss) +
                  ", validation_loss=" + str(validation_loss), end="\r")
            self.tensorboard.add_scalar('data/training_loss', training_loss, epoch)
            self.tensorboard.add_scalar('data/validation_loss', validation_loss, epoch)

            ''' early stopping '''
            if validation_loss < best_validation_loss:
                waited_epochs = 0
            else:
                if waited_epochs == patience:
                    print("Training terminated by early stopping on epoch " + str(epoch))
                    break
                waited_epochs += 1

        self.tensorboard.close()

        return training_losses, validation_losses

    def eval_metrics(self, testloader):

        correct = 0
        total = 0
        loss = 0.0
        num_batches = 0
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
                num_batches += 1

                # update numbers of total and correct predictions
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        loss /= num_batches
        return round(accuracy, 1), loss


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net(10, 0.0001, 0.01).to(device)

    print(summary(net, (3, 32, 32)))
