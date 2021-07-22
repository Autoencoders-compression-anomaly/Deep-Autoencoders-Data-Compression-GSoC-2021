import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from fastai.data import core


class AE_4D(nn.Module):
        def __init__(self, n_features=4):
            super(AE_4D, self).__init__()
            self.en1 = nn.Linear(n_features, 200)
            self.en2 = nn.Linear(200, 100)
            self.en3 = nn.Linear(100, 50)
            self.en4 = nn.Linear(50, 3)
            self.de1 = nn.Linear(3, 50)
            self.de2 = nn.Linear(50, 100)
            self.de3 = nn.Linear(100, 200)
            self.de4 = nn.Linear(200, n_features)
            self.relu = True
            if self.relu:
                self.leakyRelu = nn.LeakyReLU()
            else:
                self.tanh = nn.Tanh()

        def encode(self, x):
            if self.relu:
                return self.en4(self.leakyRelu(self.en3(self.leakyRelu(self.en2(self.leakyRelu(self.en1(x)))))))
            else:
                return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

        def decode(self, x):
            if self.relu:
                return self.de4(
                    self.leakyRelu(self.de3(self.leakyRelu(self.de2(self.leakyRelu(self.de1(self.leakyRelu(x))))))))
            else:
                return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

        def forward(self, x):
            z = self.encode(x)
            return self.decode(z)


        def describe(self):
            return '4-200-200-20-3-20-200-200-4'


class AE_24D(nn.Module):
    def __init__(self, n_features=24):
        super(AE_24D, self).__init__()
        self.en1 = nn.Linear(n_features, 200)
        self.en2 = nn.Linear(200, 100)
        self.en3 = nn.Linear(100, 50)
        self.en4 = nn.Linear(50, 15)
        self.de1 = nn.Linear(15, 50)
        self.de2 = nn.Linear(50, 100)
        self.de3 = nn.Linear(100, 200)
        self.de4 = nn.Linear(200, n_features)
        self.relu = True
        if self.relu:
            self.leakyRelu = nn.LeakyReLU()
        else:
            self.tanh = nn.Tanh()

    def encode(self, x):
        if self.relu:
            return self.en4(self.leakyRelu(self.en3(self.leakyRelu(self.en2(self.leakyRelu(self.en1(x)))))))
        else:
            return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        if self.relu:
            return self.de4(
                self.leakyRelu(self.de3(self.leakyRelu(self.de2(self.leakyRelu(self.de1(self.leakyRelu(x))))))))
        else:
            return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return '24-200-200-20-15-20-200-200-24'


class AE_19D(nn.Module):
    def __init__(self, n_features=19):
        super(AE_19D, self).__init__()
        self.en1 = nn.Linear(n_features, 200)
        self.en2 = nn.Linear(200, 100)
        self.en3 = nn.Linear(100, 50)
        self.en4 = nn.Linear(50, 15)
        self.de1 = nn.Linear(15, 50)
        self.de2 = nn.Linear(50, 100)
        self.de3 = nn.Linear(100, 200)
        self.de4 = nn.Linear(200, n_features)
        self.relu = True
        if self.relu:
            self.leakyRelu = nn.LeakyReLU()
        else:
            self.tanh = nn.Tanh()

    def encode(self, x):
        if self.relu:
            return self.en4(self.leakyRelu(self.en3(self.leakyRelu(self.en2(self.leakyRelu(self.en1(x)))))))
        else:
            return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        if self.relu:
            return self.de4(
                self.leakyRelu(self.de3(self.leakyRelu(self.de2(self.leakyRelu(self.de1(self.leakyRelu(x))))))))
        else:
            return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return '19-200-200-20-15-20-200-200-19'