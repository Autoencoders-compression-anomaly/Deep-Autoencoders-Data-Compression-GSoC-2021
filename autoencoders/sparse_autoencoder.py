from __future__ import print_function
import time
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt


class SAE(nn.Module):
    def __init__(self, n_features=24, z_dim=15):
        super(SAE, self).__init__()
        # encoder
        self.en1 = nn.Linear(n_features, 200)
        self.en2 = nn.Linear(200, 100)
        self.en3 = nn.Linear(100, 50)
        self.en4 = nn.Linear(50, z_dim)
        # decoder
        self.de1 = nn.Linear(z_dim, 50)
        self.de2 = nn.Linear(50, 100)
        self.de3 = nn.Linear(100, 200)
        self.de4 = nn.Linear(200, n_features)

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        h1 = F.leaky_relu(self.en1(x))
        h2 = F.leaky_relu(self.en2(h1))
        h3 = F.leaky_relu(self.en3(h2))
        return self.en4(h3)

    def decode(self, z):
        h4 = F.leaky_relu(self.de1(z))
        h5 = F.leaky_relu(self.de2(h4))
        h6 = F.leaky_relu(self.de3(h5))
        out = self.de4(h6)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


def sparse_loss_function(model_children, true_data, reconstructed_data, reg_param=None, evaluate=False):
    mse = nn.MSELoss()
    mse_loss = mse(reconstructed_data, true_data)

    if not evaluate:
        l1_loss = 0
        values = true_data
        for i in range(len(model_children)):
            values = F.relu((model_children[i](values)))
            l1_loss += torch.mean(torch.abs(values))

        loss = mse_loss + reg_param * l1_loss
    else:
        loss = mse_loss

    return loss


def fit(model, train_dl, train_ds, model_children, regular_param, optimizer):
    print('Training')
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(train_dl), total=int(len(train_ds) / train_dl.batch_size)):
        counter += 1
        x, _ = data
        # x = x.view(img.size(0), -1)
        optimizer.zero_grad()
        reconstructions = model(x)

        loss = sparse_loss_function(model_children=model_children, true_data=x, reconstructed_data=reconstructions,
                                    reg_param=regular_param)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / counter
    print(f" Train Loss: {loss:.6f}")

    return epoch_loss


def validate(model, test_dl, test_ds, model_children):
    print('Validating')
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dl), total=int(len(test_ds) / test_dl.batch_size)):
            counter += 1
            x, _ = data
            # x = x.view(img.size(0), -1)
            reconstructions = model(x)
            loss = sparse_loss_function(model_children=model_children, true_data=x, reconstructed_data=reconstructions,
                                        evaluate=True)
            running_loss += loss.item()

    epoch_loss = running_loss / counter
    print(f" Val Loss: {loss:.6f}")
    # save the reconstructed images every 5 epochs
    return epoch_loss


def train(variables, train_data, test_data, learning_rate, reg_param, epochs):
    sae = SAE(n_features=variables, z_dim=15)
    model_children = list(sae.children())

    # Constructs a tensor object of the data and wraps them in a TensorDataset object.
    train_ds = TensorDataset(torch.tensor(train_data.values, dtype=torch.float),
                             torch.tensor(train_data.values, dtype=torch.float))
    valid_ds = TensorDataset(torch.tensor(test_data.values, dtype=torch.float),
                             torch.tensor(test_data.values, dtype=torch.float))

    bs = 512

    # Converts the TensorDataset into a DataLoader object and combines into one DataLoaders object (a basic wrapper
    # around several DataLoader objects).
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
    # dls = core.DataLoaders(train_dl, valid_dl)

    optimizer = optim.Adam(sae.parameters(), lr=learning_rate)

    # train and validate the autoencoder neural network
    train_loss = []
    val_loss = []
    start = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss = fit(model=sae, train_dl=train_dl, train_ds=train_ds, model_children=model_children,
                               optimizer=optimizer, regular_param=reg_param)
        val_epoch_loss = validate(model=sae, test_dl=valid_dl, test_ds=valid_ds, model_children=model_children)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
    end = time.time()

    print(f"{(end - start) / 60:.3} minutes")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('D:\Desktop\GSoC-ATLAS\learning_curves\sae_loss.png')
    plt.show()

    data = torch.tensor(test_data.values, dtype=torch.float)

    pred = sae(data)
    pred = pred.detach().numpy()
    data = data.detach().numpy()

    return data, pred
