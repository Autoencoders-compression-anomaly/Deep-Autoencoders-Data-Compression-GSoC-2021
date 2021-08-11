from data_processing import preprocess_28D, preprocess_4D
import autoencoders.autoencoder as ae
import autoencoders.vae as vae
import autoencoders.sparse_autoencoder as sae
from create_plots import plot_initial_data, plot_test_pred_data, plot_4D_data, correlation_plots
from evaluate import evaluate_model
import pandas as pd
from data_loader import load_cms_data

if __name__ == "__main__":
    openCMS_data = True
    use_vae = False
    use_sae = True
    # cms_data_df = load_cms_data(filename="open_cms_data.root")
    data_df = pd.read_csv('27D_openCMS_data.csv')
    custom_norm = False
    num_of_variables = 24
    # Plot the original data
    #plot_initial_data(input_data=data_df)

    if openCMS_data:

        # Preprocess data
        data_df, train_data, test_data, scaler = preprocess_28D(data_df=data_df, num_variables=num_of_variables, custom_norm=custom_norm)
        # Plot preprocessed data
        #plot_initial_data(input_data=data_df, num_variables=num_of_variables, normalized=True)

        if use_vae:
            # Run the Variational Autoencoder and obtain the reconstructed data
            test_data, reconstructed_data = vae.train(train_data=train_data, test_data=test_data, epochs=5)
            # Plot the reconstructed along with the initial data
            plot_test_pred_data(test_data, reconstructed_data, num_variables=num_of_variables, vae=True)
        elif use_sae:
            # Run the Sparse Autoencoder and obtain the reconstructed data
            test_data, reconstructed_data = sae.train(train_data=train_data, test_data=test_data, learning_rate= 0.01, reg_param=0.1, epochs=5)
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

