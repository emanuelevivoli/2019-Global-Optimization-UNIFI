import rbfopt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

# from tensorboardX import SummaryWriter

# TODO decidere quali iperparametri ottimizzare
from net import Net

import csv



def evaluate(hyperparameters):
    lr = hyperparameters[0]   # learning rate
    # reg = hyperparameters[1]  # regularizer
    max_epochs = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(max_epochs, lr).to(device)

    # TODO allenamento modello
    training_losses = model.fit(trainloader)

    # TODO valutazione dell'accuratezza sul test set
    accuracy, test_loss = model.eval_metrics(testloader)

    # TODO salvo su file di testo l'accuratezza, insieme alle informazioni collegate (ottimizzatore, val iperparametri, loss, numero tentativo ecc.)
    # print(hyp_opt + "(lr=" + str(lr) + "), max_epochs=" + str(max_epochs) + ": accuracy=" + str(accuracy) + ", test_loss="+str(test_loss))
    
    with open('file.csv', mode='a') as file_csv:
        file_csv = csv.writer(file_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_csv.writerow([hyp_opt, str(lr), str(max_epochs), str(accuracy), str(test_loss)])

    return test_loss


with open('file.csv', mode='a') as file_csv:
    file_csv = csv.writer(file_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_csv.writerow(['hyp_opt', 'lr', 'max_epochs', 'accuracy', 'test_loss'])

# writer = SummaryWriter()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=6)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=6)

hyp_opt = "RBF"
#bb = rbfopt.RbfoptUserBlackBox(3, np.array([0] * 3), np.array([10] * 3), np.array(['R', 'I', 'R']), evaluate)
bb = rbfopt.RbfoptUserBlackBox(1, [0.00001], [0.001], ['R'], evaluate)

settings = rbfopt.RbfoptSettings(max_evaluations=50)
alg = rbfopt.RbfoptAlgorithm(settings, bb)
val, x, itercount, evalcount, fast_evalcount = alg.optimize()
