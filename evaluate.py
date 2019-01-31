import rbfopt
import numpy as np


# TODO decidere quali iperparametri ottimizzare
def evaluate(x):
    lr = x[0]   # learning rate
    reg = x[1]  # regularizer

    # TODO allenamento modello

    # TODO valutazione dell'accuratezza sul test set

    accuracy = 0.8

    # TODO salvo su file di testo l'accuratezza, insieme alle informazioni collegate (ottimizzatore, loss, numero tentativo ecc.)

    return accuracy


bb = rbfopt.RbfoptUserBlackBox(3, np.array([0] * 3), np.array([10] * 3),
                               np.array(['R', 'I', 'R']), evaluate)

settings = rbfopt.RbfoptSettings(max_evaluations=50)
alg = rbfopt.RbfoptAlgorithm(settings, bb)
val, x, itercount, evalcount, fast_evalcount = alg.optimize()
