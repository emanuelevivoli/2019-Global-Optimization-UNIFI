import sys

import torch

import rbfopt
from bayes_opt import BayesianOptimization

import csv

import utils
from net import Net

# output files
log_file = "log.txt"
csv_evaluations_file = "evaluations.csv"

# optimization parameters
max_epochs = 1
num_evaluations = 4
init_points_BAY = 2

# hyperparameters domains
hyp_domains = {"learning_rate": (0.0001, 0.1), "weight_decay": (0, 0.001)}

# gpu id
gpu = 0

def evaluate_RBF(hyperparameters):
    return -evaluate_BAY(hyperparameters[0], hyperparameters[1])


def evaluate_BAY(learning_rate, weight_decay):

    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")
    model = Net(max_epochs, learning_rate, weight_decay, hyp_opt, gpu).to(device)

    training_losses, validation_losses, training_accuracies, validation_accuracies = model.fit(trainloader, validationloader, keep_best=True, patience=-1)
    best_val_loss = min(validation_losses)
    best_epoch = validation_losses.index(best_val_loss)

    test_loss, test_accuracy = model.eval_metrics(testloader)

    with open(csv_evaluations_file, mode='a') as file_csv:
        file_csv = csv.writer(file_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_csv.writerow([hyp_opt, str(learning_rate), str(weight_decay), str(max_epochs),
                           str(training_losses[best_epoch]), str(best_val_loss), str(test_loss),
                           str(training_accuracies[best_epoch]), str(validation_accuracies[best_epoch]), str(test_accuracy)])

    return -best_val_loss


if __name__ == "__main__":

    ''' Logger creation to write on both sys.out and log file'''
    sys.stdout = utils.Logger(log_file)

    ''' CIFAR10 Dataset reading '''
    trainloader, validationloader, testloader = utils.getCIFAR10(validation=True)

    ''' Same initial point for all nets '''
    Net.same_initial_point(True)

    ''' Radial Basis Function hyperparameters optimization '''
    print("Beginning hyperparameters optimization with RBF")
    csv_header = ['hyp_opt', 'learning_rate', 'weight_decay', 'max_epochs', 'training_loss', 'validation_loss',
                  'test_loss', 'train_accuracy', 'val_accuracy', 'test_accuracy']
    with open(csv_evaluations_file, mode='a') as file_csv:
        file_csv = csv.writer(file_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_csv.writerow(csv_header)

    hyp_opt = "RBF"
    var_lower = [ hyp_domains["learning_rate"][0], hyp_domains["weight_decay"][0] ]
    var_upper = [ hyp_domains["learning_rate"][1], hyp_domains["weight_decay"][1] ]
    bb = rbfopt.RbfoptUserBlackBox(2, var_lower, var_upper, ['R', 'R'], evaluate_RBF)
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
        file_csv.writerow(csv_header)

    hyp_opt = "BAY"
    bay_opt = BayesianOptimization(f=evaluate_BAY, pbounds=hyp_domains)
    bay_opt.maximize(init_points=init_points_BAY, n_iter=num_evaluations-init_points_BAY)
    print("Results with Bayesian optimizer: " + str(bay_opt.max) + "\n")
    with open(csv_evaluations_file, mode='a') as file_csv:
        file_csv = csv.writer(file_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_csv.writerow(['', '', '', '', '', '', '', ''])
