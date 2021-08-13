from data_processing import preprocess_28D, preprocess_4D
import autoencoders.autoencoder as ae
import autoencoders.vae as vae
import autoencoders.sparse_autoencoder as sae
from create_plots import plot_initial_data, plot_test_pred_data, plot_4D_data, correlation_plots
from evaluate import evaluate_model
import pandas as pd
import argparse
from data_loader import load_cms_data


if __name__ == "__main__":
    # constructing argument parsers
    ap = argparse.ArgumentParser()
    ap.add_argument('-e', '--epochs', type=int, default=50,
                    help='number of epochs to train our autoencoder for')
    ap.add_argument('-l', '--reg_param', type=float, default=0.01,
                    help='regularization parameter `lambda`')

    ap.add_argument('-v', '--num_variables', type=int, default=24,
                    help='Number of variables we want to compress (either 19 or 24)')

    ap.add_argument('-cn', '--custom_norm', type=bool, default=False,
                    help='Whether we want to normalize all variables with min_max scaler or also use custom normalization for 4-momentum')

    ap.add_argument('-vae', '--use_vae', type=bool, default=False,
                    help='Whether to use Variational AE')
    ap.add_argument('-sae', '--use_sae', type=bool, default=True,
                    help='Whether to use Sparse AE')

    args = vars(ap.parse_args())
    epochs = args['epochs']
    reg_param = args['reg_param']
    use_vae = args['use_vae']
    use_sae = args['use_sae']
    custom_norm = args['custom_norm']
    num_of_variables = args['num_variables']
    openCMS_data = True

    # cms_data_df = load_cms_data(filename="open_cms_data.root")
    data_df = pd.read_csv('27D_openCMS_data.csv')

    lr = 0.001

    # Plot the original data
    #plot_initial_data(input_data=data_df)

    if openCMS_data:

        # Preprocess data
        data_df, train_data, test_data, scaler = preprocess_28D(data_df=data_df, num_variables=num_of_variables, custom_norm=custom_norm)
        # Plot preprocessed data
        #plot_initial_data(input_data=data_df, num_variables=num_of_variables, normalized=True)

        if use_vae:
            # Run the Variational Autoencoder and obtain the reconstructed data
            test_data, reconstructed_data = vae.train(variables=num_of_variables, train_data=train_data, test_data=test_data, epochs=5)
            # Plot the reconstructed along with the initial data
            plot_test_pred_data(test_data, reconstructed_data, num_variables=num_of_variables, vae=True)
        elif use_sae:
            # Run the Sparse Autoencoder and obtain the reconstructed data
            test_data, reconstructed_data = sae.train(variables=num_of_variables, train_data=train_data,
                                                      test_data=test_data, learning_rate=lr, reg_param=0.01, epochs=50)
            # Plot the reconstructed along with the initial data
            plot_test_pred_data(test_data, reconstructed_data, num_variables=num_of_variables, sae=True)
        else:
            # Initialize the Autoencoder
            standard_ae = ae.Autoencoder(train_data, test_data, num_variables=num_of_variables)
            # Train the standard Autoencoder and obtain the reconstructions
            test_data, reconstructed_data = standard_ae.train(test_data, epochs=30)

            # Plot the reconstructed along with the initial data
            plot_test_pred_data(test_data, reconstructed_data, num_variables=num_of_variables)

        # Evaluate the reconstructions of the network based on various metrics
        evaluate_model(y_true=test_data, y_predicted=reconstructed_data)

    else:
        # Dark machines data
        data_df, train_data, test_data = preprocess_4D('D:\Desktop\GSoC-ATLAS\data_4D.csv')

        if use_vae:
            # Run the Variational Autoencoder and obtain the reconstructed data
            test_data, reconstructed_data = vae.train(epochs=30, train_data=train_data, test_data=test_data)
            plot_4D_data(test_data, reconstructed_data)

        else:
            # Run the Autoencoder and obtain the reconstructed data
            ae_3d = ae.Autoencoder(train_data, test_data, 4)
            test_data, reconstructed_data = ae_3d.train(test_data, epochs=30)

            plot_4D_data(test_data, reconstructed_data)

