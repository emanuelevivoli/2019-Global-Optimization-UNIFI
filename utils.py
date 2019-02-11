import sys
import torch
import torchvision
from torch.utils.data import sampler
import torchvision.transforms as transforms


class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


def getCIFAR10(validation=True):

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    nw = 4      # number of workers threads
    bs = 64     # batch size

    train_size = 50000
    if validation:
        # 75% training, 25% validation
        train_size = 37500
        validation_size = 12500

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=False, num_workers=nw, sampler=ChunkSampler(train_size, 0))

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=nw)

    if validation:
        validationloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=False, num_workers=nw, sampler=ChunkSampler(validation_size, train_size))
        return trainloader, validationloader, testloader

    return trainloader, testloader

