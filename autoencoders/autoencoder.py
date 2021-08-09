import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from fastai.callback.progress import ShowGraphCallback
from torch.autograd import Variable

import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from fastai import learner
from fastai.data import core
from fastai.callback import schedule
from fastai.metrics import mse, partial
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.test_utils import *
import autoencoders.standard_autoencoders as ae

"""
This class trains the standard Autoencoder using fastai library
"""


class Autoencoder:

    def __init__(self, train, test, num_variables):
        # Constructs a tensor object of the data and wraps them in a TensorDataset object.
        train_ds = TensorDataset(torch.tensor(train.values, dtype=torch.float),
                                 torch.tensor(train.values, dtype=torch.float))
        valid_ds = TensorDataset(torch.tensor(test.values, dtype=torch.float),
                                 torch.tensor(test.values, dtype=torch.float))

        bs = 512

        # Converts the TensorDataset into a DataLoader object and combines into one DataLoaders object (a basic wrapper
        # around several DataLoader objects).
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
        valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
        self.dls = core.DataLoaders(train_dl, valid_dl)

        # define our model
        if num_variables == 4:
            self.model = ae.AE_4D()
        elif num_variables == 24:
            self.model = ae.AE_24D()
        else:
            self.model = ae.AE_19D()

    def train(self, test_set, epochs):
        loss_function = nn.MSELoss()

        weight_decay = 1e-6

        recorder = learner.Recorder()
        learn = learner.Learner(self.dls, model=self.model, wd=weight_decay, loss_func=loss_function, cbs=recorder)
        plt.figure()
        lr_min, lr_steep = learn.lr_find()
        # plt.show()

        print('Learning rate with the minimum loss:', lr_min)
        print('Learning rate with the steepest gradient:', lr_steep)

        start = time.perf_counter()  # Starts timer
        # train our autoencoder
        learn.fit_one_cycle(epochs, lr_min, cbs=[ShowGraphCallback()])
        end = time.perf_counter()  # Ends timer
        delta_t = end - start
        print('Training took', delta_t, 'seconds')

        plt.figure()
        recorder.plot_loss()
        plt.show()

        data = torch.tensor(test_set.values, dtype=torch.float)

        pred = self.model(data)
        pred = pred.detach().numpy()
        data = data.detach().numpy()

        return data, pred
