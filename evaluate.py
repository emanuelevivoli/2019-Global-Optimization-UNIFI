import torch
import torchvision
import torchvision.transforms as transforms

import rbfopt
from bayes_opt import BayesianOptimization
# from tensorboardX import SummaryWriter

import csv

import utils
from net import Net

# TODO iperparametri ottimizati: learning_rate, weight_decay (regolarizzatore). Vedere se aggiungerne altri


def evaluate_RBF(hyperparameters):
    return evaluate_BAY(hyperparameters[0], hyperparameters[1])


def evaluate_BAY(learning_rate, weight_decay):
    max_epochs = 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(max_epochs, learning_rate, weight_decay).to(device)

    patience = 1
    training_losses, validation_losses = model.fit(trainloader, validationloader, patience)

    accuracy, test_loss = model.eval_metrics(testloader)

    with open('file.csv', mode='a') as file_csv:
        file_csv = csv.writer(file_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_csv.writerow([hyp_opt, str(learning_rate), str(weight_decay), str(max_epochs), str(training_losses[-1]),
                           str(validation_losses[-1]), str(test_loss), str(accuracy)])

    print(hyp_opt + "(lr=" + str(learning_rate) + ", wd=" + str(weight_decay) + "), max_epochs=" + str(max_epochs) +
          ": training_loss=" + str(training_losses[-1]) + ", validation_loss=" + str(validation_losses[-1]) +
          "test_loss=" + str(test_loss) + ", test_accuracy=" + str(accuracy))

    return test_loss


if __name__ == "__main__":
    # writer = SummaryWriter()
    ''' CIFAR10 Dataset reading '''
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainloader, validationloader, testloader = utils.getCIFAR10(transform=transform, validation=True)

    ''' Radial Basis Function hyperparameters optimization '''
    with open('file.csv', mode='a') as file_csv:
        file_csv = csv.writer(file_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_csv.writerow(['hyp_opt', 'learning_rate', 'weight_decay', 'max_epochs', 'training_loss', 'validation_loss', 'test_loss', 'accuracy'])

    hyp_opt = "RBF"
    bb = rbfopt.RbfoptUserBlackBox(2, [0.00001, 0.0], [0.001, 0.001], ['R', 'R'], evaluate_RBF)
    settings = rbfopt.RbfoptSettings(max_evaluations=10)
    alg = rbfopt.RbfoptAlgorithm(settings, bb)
    val, x, itercount, evalcount, fast_evalcount = alg.optimize()

    ''' Bayesian hyperparameters optimization '''
    with open('file.csv', mode='a') as file_csv:
        file_csv = csv.writer(file_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_csv.writerow(['hyp_opt', 'learning_rate', 'weight_decay', 'max_epochs', 'training_loss', 'validation_loss', 'test_loss', 'accuracy'])

    hyp_opt = "BAY"
    pb = {"learning_rate": (0.00001, 0.001), "weight_decay": (0, 0.001)}
    bay_opt = BayesianOptimization(f=evaluate_BAY, pbounds=pb)
    bay_opt.maximize(init_points=5, n_iter=5)
    print(bay_opt.max)
