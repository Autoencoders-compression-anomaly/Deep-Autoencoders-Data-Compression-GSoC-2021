from fastai.vision import *
from autoencoders.variational_autoencoder import *


def train(variables, train_data, test_data, epochs):
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
    dls = core.DataLoaders(train_dl, valid_dl)

    vae = VAE(n_features=variables, z_dim=15)

    weight_decay = 1e-6
    recorder = learner.Recorder()
    learn = learner.Learner(dls, model=vae, wd=weight_decay, loss_func=vae.vae_loss_function, cbs=recorder)
    plt.figure()
    lr_min, lr_steep = learn.lr_find()

    print('Learning rate with the minimum loss:', lr_min)
    print('Learning rate with the steepest gradient:', lr_steep)

    start = time.perf_counter()  # Starts timer
    # train our autoencoder
    learn.fit_one_cycle(epochs, 0.01, cbs=[ShowGraphCallback()])
    end = time.perf_counter()  # Ends timer
    delta_t = end - start
    print('Training took', delta_t, 'seconds')

    plt.figure()
    recorder.plot_loss()
    plt.show()

    data = torch.tensor(test_data.values, dtype=torch.float)

    pred = vae(data)
    pred = pred.detach().numpy()
    data = data.detach().numpy()

    return data, pred
