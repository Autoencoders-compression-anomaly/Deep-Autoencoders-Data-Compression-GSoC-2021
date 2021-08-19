from data_processing import preprocess_28D
import autoencoders.autoencoder as ae
import autoencoders.variational_autoencoder as vae
import autoencoders.sparse_autoencoder as sae
from create_plots import plot_initial_data, plot_test_pred_data, correlation_plot
from evaluate import evaluate_model
import pandas as pd
import argparse
from data_loader import load_cms_data


if __name__ == "__main__":
    # constructing argument parsers
    ap = argparse.ArgumentParser()
    ap.add_argument('-e', '--epochs', type=int, default=50,
                    help='number of epochs to train our autoencoder for')

    ap.add_argument('-v', '--num_variables', type=int, default=24,
                    help='Number of variables we want to compress (either 19 or 24)')

    ap.add_argument('-cn', '--custom_norm', type=bool, default=False,
                    help='Whether we want to normalize all variables with min_max scaler or also use custom normalization for 4-momentum')

    ap.add_argument('-vae', '--use_vae', type=bool, default=False,
                    help='Whether to use Variational AE')
    ap.add_argument('-sae', '--use_sae', type=bool, default=False,
                    help='Whether to use Sparse AE')
    ap.add_argument('-l1', '--l1', type=bool, default=False,
                    help='Whether to use L1 loss or KL-divergence in the Sparse AE')
    ap.add_argument('-p', '--plot', type=bool, default=False,
                    help='Whether to use L1 loss or KL-divergence in the Sparse AE')

    args = vars(ap.parse_args())
    epochs = args['epochs']
    use_vae = args['use_vae']
    use_sae = args['use_sae']
    custom_norm = args['custom_norm']
    num_of_variables = args['num_variables']
    create_plots = args['plot']
    l1 = args['l1']
    reg_param = 0.001
    # sparsity parameter for KL loss in SAE
    RHO = 0.05
    # learning rate
    lr = 0.001

    cms_data_df = load_cms_data(filename="open_cms_data.root")
    data_df = pd.read_csv('27D_openCMS_data.csv')

    if create_plots:
        # Plot the original data
        plot_initial_data(input_data=data_df, num_variables=num_of_variables)

        # Plot correlation matrix between the input variables of the data
        correlation_plot(data_df)

    # Preprocess data
    data_df, train_data, test_data, scaler = preprocess_28D(data_df=data_df, num_variables=num_of_variables, custom_norm=custom_norm)

    if create_plots:
        # Plot preprocessed data
        plot_initial_data(input_data=data_df, num_variables=num_of_variables, normalized=True)

    if use_vae:
        # Run the Variational Autoencoder and obtain the reconstructed data
        test_data, reconstructed_data = vae.train(variables=num_of_variables, train_data=train_data, test_data=test_data, epochs=epochs, learning_rate=lr)
        # Plot the reconstructed along with the initial data
        plot_test_pred_data(test_data, reconstructed_data, num_variables=num_of_variables, vae=True)
    elif use_sae:
        # Run the Sparse Autoencoder and obtain the reconstructed data
        test_data, reconstructed_data = sae.train(variables=num_of_variables, train_data=train_data,
                                                  test_data=test_data, learning_rate=lr, reg_param=reg_param, epochs=epochs, RHO=RHO, l1=l1)
        # Plot the reconstructed along with the initial data
        plot_test_pred_data(test_data, reconstructed_data, num_variables=num_of_variables, sae=True)
    else:
        # Initialize the Autoencoder
        standard_ae = ae.Autoencoder(train_data, test_data, num_variables=num_of_variables)
        # Train the standard Autoencoder and obtain the reconstructions
        test_data, reconstructed_data = standard_ae.train(test_data, epochs=epochs)

        # Plot the reconstructed along with the initial data
        plot_test_pred_data(test_data, reconstructed_data, num_variables=num_of_variables)

    # Evaluate the reconstructions of the network based on various metrics
    evaluate_model(y_true=test_data, y_predicted=reconstructed_data)