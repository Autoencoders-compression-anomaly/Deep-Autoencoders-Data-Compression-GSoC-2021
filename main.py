from data_processing import preprocess_28D, preprocess_4D
import autoencoders.standard_AE as ae
from create_plots import plot_initial_data, plot_test_pred_data, plot_4D_data
import pandas as pd
from data_loader import load_cms_data

if __name__ == "__main__":
    #cms_data_df = load_cms_data(filename="open_cms_data.root")
    data_df = pd.read_csv('27D_openCMS_data.csv')

    # Plot the original data
    plot_initial_data(data_df)

    # Preprocessing
    data_df, train_data, test_data = preprocess_28D(data_df)

    # Plot the normalized data
    plot_initial_data(data_df)

    # Run the Autoencoder and obtain the reconstructed data
    standard_ae = ae.Standard_Autoencoder(input_dim=28, z_dim=20)
    reconstructed_data = standard_ae.train(train_data=train_data, test_data=test_data)
    # Plot the original along with the reconstructed data
    plot_test_pred_data(test_data, reconstructed_data)
