from data_processing import preprocess_28D, preprocess_4D
import autoencoders.autoencoder as ae
import autoencoders.variational_autoencoder as vae
from create_plots import plot_initial_data, plot_test_pred_data, plot_4D_data, plot_residuals, correlation_plots
import pandas as pd
from data_loader import load_cms_data

if __name__ == "__main__":
    data_28D = True
    use_vae = True
    # cms_data_df = load_cms_data(filename="open_cms_data.root")
    data_df = pd.read_csv('27D_openCMS_data.csv')

    # Plot the original data
    plot_initial_data(input_data=data_df)

    if data_28D:
        data_df, train_data, test_data = preprocess_28D(data_df=data_df, data_4D=False, num_variables=19)
        #plot_initial_data(input_data=data_df, normalized=True)

        # Run the Autoencoder and obtain the reconstructed data
        ae_28d = ae.Autoencoder(train_data, test_data, 19)

        test_data, reconstructed_data = ae_28d.train(test_data)

        plot_test_pred_data(test_data, reconstructed_data, 19)

    else:
        # Open CMS data
        #data_df, train_data, test_data = preprocess_28D(data_df=data_df, data_4D=True, num_variables=4)
        # Dark machines data
        data_df, train_data, test_data = preprocess_4D('D:\Desktop\GSoC-ATLAS\data_4D.csv')

        if use_vae:
            # Run the Variational Autoencoder and obtain the reconstructed data
            test_data, reconstructed_data = vae.train(epochs=50, train_data=train_data, test_data=test_data)
            plot_4D_data(test_data, reconstructed_data)

        else:
            # Run the Autoencoder and obtain the reconstructed data
            ae_3d = ae.Autoencoder(train_data, test_data, 4)
            test_data, reconstructed_data = ae_3d.train(test_data)

            plot_4D_data(test_data, reconstructed_data)


    """
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
    """
