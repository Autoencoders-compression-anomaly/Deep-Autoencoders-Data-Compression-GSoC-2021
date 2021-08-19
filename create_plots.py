import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="white")


def plot_initial_data(input_data, num_variables, normalized=False):
    input_data = input_data.sort_values(by=['ak5PFJets_pt_'])

    if normalized:
        save_dir = "D:\Desktop\GSoC-ATLAS\preprocessed_data_plots"
    else:
        save_dir = "D:\Desktop\GSoC-ATLAS\initial_data_plots"

    prefix = 'ak5PFJets_'

    if num_variables == 24:
        save_dir = "D:\Desktop\preprocessed_data_plots\d24"

        variable_list = ['pt_', 'eta_', 'phi_', 'mass_', 'mJetArea',
                         'mChargedHadronEnergy', 'mNeutralHadronEnergy',
                         'mPhotonEnergy',
                         'mElectronEnergy', 'mMuonEnergy', 'mHFHadronEnergy',
                         'mHFEMEnergy', 'mChargedHadronMultiplicity',
                         'mNeutralHadronMultiplicity',
                         'mPhotonMultiplicity', 'mElectronMultiplicity',
                         'mMuonMultiplicity',
                         'mHFHadronMultiplicity', 'mHFEMMultiplicity', 'mChargedEmEnergy',
                         'mChargedMuEnergy', 'mNeutralEmEnergy', 'mChargedMultiplicity',
                         'mNeutralMultiplicity']

        branches = [prefix + 'pt_', prefix + 'eta_', prefix + 'phi_', prefix + 'mass_',
                    prefix + 'mJetArea', prefix + 'mChargedHadronEnergy', prefix + 'mNeutralHadronEnergy',
                    prefix + 'mPhotonEnergy',
                    prefix + 'mElectronEnergy', prefix + 'mMuonEnergy', prefix + 'mHFHadronEnergy',
                    prefix + 'mHFEMEnergy', prefix + 'mChargedHadronMultiplicity',
                    prefix + 'mNeutralHadronMultiplicity',
                    prefix + 'mPhotonMultiplicity', prefix + 'mElectronMultiplicity',
                    prefix + 'mMuonMultiplicity',
                    prefix + 'mHFHadronMultiplicity', prefix + 'mHFEMMultiplicity', prefix + 'mChargedEmEnergy',
                    prefix + 'mChargedMuEnergy', prefix + 'mNeutralEmEnergy', prefix + 'mChargedMultiplicity',
                    prefix + 'mNeutralMultiplicity']
    else:
        save_dir = "D:\Desktop\GSoC-ATLAS\preprocessed_data_plots\d19"

        variable_list = ['pt_', 'eta_', 'phi_', 'mass_', 'mJetArea',
                         'mChargedHadronEnergy', 'mNeutralHadronEnergy',
                         'mPhotonEnergy', 'mHFHadronEnergy',
                         'mHFEMEnergy', 'mChargedHadronMultiplicity',
                         'mNeutralHadronMultiplicity',
                         'mPhotonMultiplicity', 'mElectronMultiplicity',
                         'mHFHadronMultiplicity', 'mHFEMMultiplicity', 'mNeutralEmEnergy', 'mChargedMultiplicity',
                         'mNeutralMultiplicity']

        branches = [prefix + 'pt_', prefix + 'eta_', prefix + 'phi_', prefix + 'mass_',
                    prefix + 'mJetArea', prefix + 'mChargedHadronEnergy', prefix + 'mNeutralHadronEnergy',
                    prefix + 'mPhotonEnergy', prefix + 'mHFHadronEnergy',
                    prefix + 'mHFEMEnergy', prefix + 'mChargedHadronMultiplicity',
                    prefix + 'mNeutralHadronMultiplicity',
                    prefix + 'mPhotonMultiplicity', prefix + 'mElectronMultiplicity',
                    prefix + 'mHFHadronMultiplicity', prefix + 'mHFEMMultiplicity',
                    prefix + 'mNeutralEmEnergy', prefix + 'mChargedMultiplicity',
                    prefix + 'mNeutralMultiplicity']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_bins = 100
    save = True  # Option to save figure

    for kk in range(0, num_variables):
        if branches[kk] == prefix + 'pt_' or branches[kk] == prefix + 'mass_':
            n_hist_data, bin_edges, _ = plt.hist(input_data[branches[kk]], color='orange', label='Input', alpha=1,
                                                 bins=n_bins, log=True)
            plt.xlabel(xlabel=variable_list[kk])
            plt.ylabel('# of jets')
            plt.suptitle(variable_list[kk])
            if save:
                plt.savefig(os.path.join(save_dir, variable_list[kk] + '.png'))
        elif branches[kk] == prefix + 'phi_':
            n_hist_data, bin_edges, _ = plt.hist(input_data[branches[kk]], color='orange', label='Input', alpha=1,
                                                 bins=50)
            plt.xlabel(xlabel=variable_list[kk])
            plt.ylabel('# of jets')
            plt.suptitle(variable_list[kk])
            if save:
                plt.savefig(os.path.join(save_dir, variable_list[kk] + '.png'))
        else:
            n_hist_data, bin_edges, _ = plt.hist(input_data[branches[kk]], color='orange', label='Input', alpha=1,
                                                 bins=n_bins)
            plt.xlabel(xlabel=variable_list[kk])
            plt.ylabel('# of jets')
            plt.suptitle(variable_list[kk])
            if save:
                plt.savefig(os.path.join(save_dir, variable_list[kk] + '.png'))
        plt.figure()
        plt.show()


def plot_test_pred_data(test_data, predicted_data, num_variables, vae=False, sae=False):
    if num_variables == 24:
        if sae:
            save_dir = "D:\Desktop\GSoC-ATLAS\SAE_plots\d24"
        elif vae:
            save_dir = "D:\Desktop\GSoC-ATLAS\VAE_plots\d24"
        else:
            save_dir = "D:\Desktop\GSoC-ATLAS\AE_plots\d24"

        variable_list = ['pt_', 'eta_', 'phi_', 'mass_', 'mJetArea',
                         'mChargedHadronEnergy', 'mNeutralHadronEnergy',
                         'mPhotonEnergy',
                         'mElectronEnergy', 'mMuonEnergy', 'mHFHadronEnergy',
                         'mHFEMEnergy', 'mChargedHadronMultiplicity',
                         'mNeutralHadronMultiplicity',
                         'mPhotonMultiplicity', 'mElectronMultiplicity',
                         'mMuonMultiplicity',
                         'mHFHadronMultiplicity', 'mHFEMMultiplicity', 'mChargedEmEnergy',
                         'mChargedMuEnergy', 'mNeutralEmEnergy', 'mChargedMultiplicity',
                         'mNeutralMultiplicity']
    else:
        if sae:
            save_dir = "D:\Desktop\GSoC-ATLAS\SAE_plots\d19"
        elif vae:
            save_dir = "D:\Desktop\GSoC-ATLAS\VAE_plots\d19"
        else:
            save_dir = "D:\Desktop\GSoC-ATLAS\AE_plots\d19"

        variable_list = ['pt_', 'eta_', 'phi_', 'mass_', 'mJetArea',
                         'mChargedHadronEnergy', 'mNeutralHadronEnergy',
                         'mPhotonEnergy', 'mHFHadronEnergy',
                         'mHFEMEnergy', 'mChargedHadronMultiplicity',
                         'mNeutralHadronMultiplicity',
                         'mPhotonMultiplicity', 'mElectronMultiplicity',
                         'mHFHadronMultiplicity', 'mHFEMMultiplicity', 'mNeutralEmEnergy', 'mChargedMultiplicity',
                         'mNeutralMultiplicity']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    colors = ['pink', 'green']
    n_bins = 100
    save = True  # Option to save figure

    # plot the input data along with the reconstructed from the AE
    for kk in np.arange(num_variables):
        plt.figure()
        n_hist_data, bin_edges, _ = plt.hist(test_data[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
        n_hist_pred, _, _ = plt.hist(predicted_data[:, kk], color=colors[0], label='Output', alpha=0.8, bins=bin_edges)
        plt.suptitle(variable_list[kk])
        plt.xlabel(xlabel=variable_list[kk])
        plt.ylabel('Number of jets')
        plt.yscale('log')
        plt.legend()
        if save:
            plt.savefig(os.path.join(save_dir, variable_list[kk] + '.png'))

        plt.show()


def plot_4D_data(test_data, predicted_data):
    save_dir = "D:\Desktop\GSoC-ATLAS\AE_4D_plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    variable_list = [r'$E$', r'$p_T$', r'$\eta$', r'$\phi$']
    colors = ['pink', 'green']

    alph = 0.8
    n_bins = 200

    for kk in np.arange(4):
        plt.figure()
        n_hist_data, bin_edges, _ = plt.hist(test_data[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
        n_hist_pred, _, _ = plt.hist(predicted_data[:, kk], color=colors[0], label='Output', alpha=alph, bins=bin_edges)
        plt.suptitle(variable_list[kk])
        plt.xlabel(xlabel=variable_list[kk])
        plt.ylabel('Number of events')

        plt.yscale('log')
        plt.show()


def plot_residuals(test_data, predicted_data):
    # Calculate the residuals
    residual_test = np.absolute(test_data - predicted_data)
    # residual_train = np.absolute(train_data - prediction_train)
    plt.figure()

    # Plotting the scatter plots
    print("These are the scatter plots")
    plt.scatter(test_data, residual_test)
    plt.title("Test data")
    plt.show()

    plt.figure()
    # Plotting Histograms
    print("These are the histograms")
    plt.hist(residual_test, 50)
    plt.title("Residuals on test data")
    plt.show()


def correlation_plot(data):
    data = data.drop(['entry', 'subentry'], axis=1)
    data.columns = data.columns.str.lstrip("ak5PFJets")
    data.columns = data.columns.str.lstrip("_")

    # Compute the correlation matrix
    corr = data.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()
