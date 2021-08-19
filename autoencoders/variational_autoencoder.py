from __future__ import print_function
import time
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm


class VAE(nn.Module):
    def __init__(self, n_features=24, z_dim=15):
        super(VAE, self).__init__()

        self.en1 = nn.Linear(n_features, 200)
        self.en2 = nn.Linear(200, 100)
        self.en3 = nn.Linear(100, 50)

        # distribution parameters
        self.fc_mu = nn.Linear(50, z_dim)
        self.fc_logvar = nn.Linear(50, z_dim)

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
        return self.fc_mu(h3), self.fc_logvar(h3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = F.leaky_relu(self.de1(z))
        h5 = F.leaky_relu(self.de2(h4))
        h6 = F.leaky_relu(self.de3(h5))
        out = self.de4(h6)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.n_features))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss_function(recon_x, x, logvar, mu):
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon_x, x)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = MSE + KLD
    return loss


def fit(model, train_dl, train_ds, optimizer):
    print('Training')
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(train_dl), total=int(len(train_ds) / train_dl.batch_size)):
        counter += 1
        x, _ = data
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(x)

        loss = vae_loss_function(recon_x=reconstruction, x=x, mu=mu, logvar=logvar)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / counter
    print(f" Train Loss: {loss:.6f}")

    return epoch_loss


def validate(model, test_dl, test_ds):
    print('Validating')
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dl), total=int(len(test_ds) / test_dl.batch_size)):
            counter += 1
            x, _ = data
            reconstruction, mu, logvar = model(x)

            loss = vae_loss_function(recon_x=reconstruction, x=x, mu=mu, logvar=logvar)

            running_loss += loss.item()

    epoch_loss = running_loss / counter
    print(f" Val Loss: {loss:.6f}")
    # save the reconstructed images every 5 epochs
    return epoch_loss


def train(variables, train_data, test_data, learning_rate, epochs):
    sae = VAE(n_features=variables, z_dim=15)

    # Constructs a tensor object of the data and wraps them in a TensorDataset object.
    train_ds = TensorDataset(torch.tensor(train_data.values, dtype=torch.float),
                             torch.tensor(train_data.values, dtype=torch.float))
    valid_ds = TensorDataset(torch.tensor(test_data.values, dtype=torch.float),
                             torch.tensor(test_data.values, dtype=torch.float))

    bs = 512

    # Converts the TensorDataset into a DataLoader object
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

    optimizer = optim.Adam(sae.parameters(), lr=learning_rate)

    # train and validate the autoencoder neural network
    train_loss = []
    val_loss = []
    start = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss = fit(model=sae, train_dl=train_dl, train_ds=train_ds,
                               optimizer=optimizer)
        val_epoch_loss = validate(model=sae, test_dl=valid_dl, test_ds=valid_ds)
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
    pred = pred[0].detach().numpy()
    data = data.detach().numpy()

    return data, pred
