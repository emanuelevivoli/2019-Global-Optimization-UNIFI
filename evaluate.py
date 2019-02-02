import torch
import torchvision
import torchvision.transforms as transforms

import rbfopt
from bayes_opt import BayesianOptimization

# TODO iperparametri ottimizati: learning_rate, weight_decay (regolarizzatore). Vedere se aggiungerne altri
from net import Net


def evaluate_RBF(hyperparameters):
    return evaluate_BAY(hyperparameters[0], hyperparameters[1])

def evaluate_BAY(learning_rate, weight_decay):
    max_epochs = 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(max_epochs, learning_rate, weight_decay).to(device)

    training_losses = model.fit(trainloader)

    accuracy, test_loss = model.eval_metrics(testloader)

    # TODO salvo su file di testo l'accuratezza, insieme alle informazioni collegate (ottimizzatore, val iperparametri, loss, numero tentativo ecc.)
    print(hyp_opt + "(lr=" + str(learning_rate) + "), max_epochs=" + str(max_epochs) + ": train_loss=" + str(training_losses[-1]) +
          ", test_loss=" + str(test_loss) + ", test_accuracy=" + str(accuracy))

    return test_loss


''' CIFAR10 Dataset reading'''
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=6)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=6)


''' Radial Basis Function hyperparameters optimization '''
hyp_opt = "RBF"
bb = rbfopt.RbfoptUserBlackBox(2, [0.00001, 0.0], [0.001, 0.001], ['R', 'R'], evaluate_RBF)
settings = rbfopt.RbfoptSettings(max_evaluations=4)
alg = rbfopt.RbfoptAlgorithm(settings, bb)
val, x, itercount, evalcount, fast_evalcount = alg.optimize()


''' Bayesian hyperparameters optimization '''
hyp_opt = "BAY"
pb = {"learning_rate": (0.00001, 0.001), "weight_decay": (0, 0.001)}
bay_opt = BayesianOptimization(f=evaluate_BAY, pbounds=pb)
bay_opt.maximize(init_points=2, n_iter=4)
print(bay_opt.max)
