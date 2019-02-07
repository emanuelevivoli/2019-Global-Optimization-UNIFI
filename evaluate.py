import sys

import torch
import torchvision.transforms as transforms

import rbfopt
from bayes_opt import BayesianOptimization

import csv

import utils
from net import Net

log_file = "log.txt"
csv_evaluations_file = "evaluations.csv"
max_epochs = 60
num_evaluations = 25
init_points_BAY = 5


def evaluate_RBF(hyperparameters):
    return -evaluate_BAY(hyperparameters[0], hyperparameters[1])


def evaluate_BAY(learning_rate, weight_decay):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(max_epochs, learning_rate, weight_decay).to(device)

    patience = 1
    training_losses, validation_losses = model.fit(trainloader, validationloader, patience)

    accuracy, test_loss = model.eval_metrics(testloader)

    with open(csv_evaluations_file, mode='a') as file_csv:
        file_csv = csv.writer(file_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_csv.writerow([hyp_opt, str(learning_rate), str(weight_decay), str(max_epochs), str(training_losses[-1]),
                           str(validation_losses[-1]), str(test_loss), str(accuracy)])

    return -validation_losses[-1]


if __name__ == "__main__":

    ''' Logger creation to write on both sys.out and log file'''
    sys.stdout = utils.Logger(log_file)

    ''' CIFAR10 Dataset reading '''
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainloader, validationloader, testloader = utils.getCIFAR10(transform=transform, validation=True)

    ''' Radial Basis Function hyperparameters optimization '''
    print("Beginning hyperparameters optimization with RBF")
    with open(csv_evaluations_file, mode='a') as file_csv:
        file_csv = csv.writer(file_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_csv.writerow(['hyp_opt', 'learning_rate', 'weight_decay', 'max_epochs', 'training_loss', 'validation_loss', 'test_loss', 'accuracy'])

    hyp_opt = "RBF"
    bb = rbfopt.RbfoptUserBlackBox(2, [0.0001, 0.0], [0.1, 0.001], ['R', 'R'], evaluate_RBF)
    settings = rbfopt.RbfoptSettings(max_evaluations=num_evaluations, target_objval=0.0)
    alg = rbfopt.RbfoptAlgorithm(settings, bb)
    val, x, itercount, evalcount, fast_evalcount = alg.optimize()
    print("Results with RBF optimizer: " + str({"target": val, "learning_rate": x[0], "weight_decay": x[1]}) + "\n")
    with open(csv_evaluations_file, mode='a') as file_csv:
        file_csv = csv.writer(file_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_csv.writerow(['', '', '', '', '', '', '', ''])

    ''' Bayesian hyperparameters optimization '''
    print("Beginning hyperparameters optimization with BAY")
    with open(csv_evaluations_file, mode='a') as file_csv:
        file_csv = csv.writer(file_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_csv.writerow(['hyp_opt', 'learning_rate', 'weight_decay', 'max_epochs', 'training_loss', 'validation_loss', 'test_loss', 'accuracy'])

    hyp_opt = "BAY"
    pb = {"learning_rate": (0.0001, 0.1), "weight_decay": (0, 0.001)}
    bay_opt = BayesianOptimization(f=evaluate_BAY, pbounds=pb)
    bay_opt.maximize(init_points=init_points_BAY, n_iter=num_evaluations-init_points_BAY)
    print("Results with Bayesian optimizer: " + str(bay_opt.max) + "\n")
    with open(csv_evaluations_file, mode='a') as file_csv:
        file_csv = csv.writer(file_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_csv.writerow(['', '', '', '', '', '', '', ''])
