from data_processing import preprocess_28D, preprocess_4D
import autoencoders.standard_AE as ae
import autoencoders.variational_AE as vae
from create_plots import plot_initial_data, plot_test_pred_data, plot_4D_data, plot_residuals, correlation_plots
import pandas as pd
from data_loader import load_cms_data

if __name__ == "__main__":
    #cms_data_df = load_cms_data(filename="open_cms_data.root")
    data_df = pd.read_csv('27D_openCMS_data.csv')

    # Plot the original data
    plot_initial_data(input_data=data_df)

    # Preprocessing
    data_df, train_data, test_data = preprocess_28D(data_df=data_df)

    # Plot the normalized data
    plot_initial_data(input_data=data_df, normalized=True)


    # Run the Autoencoder and obtain the reconstructed data
    standard_ae = ae.Standard_Autoencoder(input_dim=28, z_dim=20)
    reconstructed_data = standard_ae.train(train_data=train_data, test_data=test_data)
    reconstructed_data_df = pd.DataFrame(reconstructed_data)

    prefix = 'ak5PFJets_'
    # Set column names
    reconstructed_data_df.columns = [prefix + 'pt_', prefix + 'eta_', prefix + 'phi_', prefix + 'mass_',
                                     prefix + 'fX', prefix + 'fY', prefix + 'fZ', prefix + 'mJetArea',
                                     prefix + 'mPileupEnergy',
                                     prefix + 'mChargedHadronEnergy', prefix + 'mNeutralHadronEnergy', prefix + 'mPhotonEnergy',
                                     prefix + 'mElectronEnergy', prefix + 'mMuonEnergy', prefix + 'mHFHadronEnergy',
                                     prefix + 'mHFEMEnergy', prefix + 'mChargedHadronMultiplicity',
                                     prefix + 'mNeutralHadronMultiplicity',
                                     prefix + 'mPhotonMultiplicity', prefix + 'mElectronMultiplicity',
                                     prefix + 'mMuonMultiplicity',
                                     prefix + 'mHFHadronMultiplicity', prefix + 'mHFEMMultiplicity', prefix + 'mChargedEmEnergy',
                                     prefix + 'mChargedMuEnergy', prefix + 'mNeutralEmEnergy', prefix + 'mChargedMultiplicity',
                                     prefix + 'mNeutralMultiplicity']


    # Plot the original along with the reconstructed data
    plot_test_pred_data(test_data, reconstructed_data)

    # Plot the residuals
    plot_residuals(test_data, reconstructed_data)

    # Plot the correlations
    correlation_plots(test_data, reconstructed_data_df)

    # Run the Autoencoder and obtain the reconstructed data
    #variational_ae = vae.Variational_Autoencoder(input_dim=28, z_dim=20)
    #reconstructed_data = variational_ae.train(train_data=train_data, test_data=test_data)

    # Plot the original along with the reconstructed data
    #plot_test_pred_data(test_data=test_data, predicted_data=reconstructed_data, vae=True)
