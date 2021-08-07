from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset


class VAE(nn.Module):
    def __init__(self, n_features=4, z_dim=3):
        super(VAE, self).__init__()

        self.en1 = nn.Linear(n_features, 200)
        self.en2 = nn.Linear(200, 100)
        self.en3 = nn.Linear(100, 50)
        self.en4 = nn.Linear(50, z_dim)
        self.de1 = nn.Linear(z_dim, 50)
        self.de2 = nn.Linear(50, 100)
        self.de3 = nn.Linear(100, 200)
        self.de4 = nn.Linear(200, n_features)

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        h1 = F.relu(self.en1(x))
        h2 = F.relu(self.en2(h1))
        h3 = F.relu(self.en3(h2))
        return self.en4(h3), self.en4(h3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = F.relu(self.de1(z))
        h5 = F.relu(self.de2(h4))
        h6 = F.relu(self.de3(h5))
        return self.de4(h6)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.n_features))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()
optimizer = optim.Adam(model.parameters(), lr=0.003)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 4), reduction='sum')
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD


def train(epochs, train_data, test_data):
    # Constructs a tensor object of the data and wraps them in a TensorDataset object.
    train_ds = TensorDataset(torch.tensor(train_data.values, dtype=torch.float),
                             torch.tensor(train_data.values, dtype=torch.float))
    valid_ds = TensorDataset(torch.tensor(test_data.values, dtype=torch.float),
                             torch.tensor(test_data.values, dtype=torch.float))

    bs = 256

    # Converts the TensorDataset into a DataLoader object and combines into one DataLoaders object (a basic wrapper
    # around several DataLoader objects).
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_dl):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_dl),
                           100. * batch_idx / len(train_dl),
                           loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_dl)))

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(valid_dl):
                recon_batch, mu, logvar = model(data)
                test_loss += loss_function(recon_batch, data, mu, logvar).item()

        test_loss /= len(valid_dl)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    data = torch.tensor(test_data.values, dtype=torch.float)

    pred = model(data)[0]
    pred = pred.detach().numpy()
    data = data.detach().numpy()

    return data, pred
