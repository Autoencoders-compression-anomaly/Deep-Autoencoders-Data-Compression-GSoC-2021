import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import corner
#import arviz as az


def plot_initial_data(input_data, normalized=False):
    input_data = input_data.sort_values(by=['ak5PFJets_pt_'])

    if normalized:
        save_dir = "D:\Desktop\GSoC-ATLAS\preprocessed_data_plots"
    else:
        save_dir = "D:\Desktop\GSoC-ATLAS\initial_data_plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

    prefix = 'ak5PFJets_'
    n_bins = 100
    save = True  # Option to save figure

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

    for kk in range(0, 24):
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
        plt.show()


def plot_test_pred_data(test_data, predicted_data, num_variables, vae=False):
    if vae:
        save_dir = "D:\Desktop\GSoC-ATLAS\VAE_plots"


    if num_variables == 24:
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
    prefix = 'ak5PFJets_'
    n_bins = 100
    save = True  # Option to save figure

    #predicted_data = predicted_data.detach().numpy()
    #test_data = test_data.values

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


# plot(data_df = pd.read_csv('27D_openCMS_preprocessed_data.csv'))

def plot_4D_data(test_data, predicted_data):

    save_dir = "D:\Desktop\GSoC-ATLAS\AE_4D_plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save = True

    variable_list = [r'$E$', r'$p_T$', r'$\eta$', r'$\phi$']
    colors = ['pink', 'green']

    #test_data = test_data.values

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
    #residual_train = np.absolute(train_data - prediction_train)
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


def correlation_plots(test_data, predicted_data):
    pt = 'ak5PFJets_pt_'
    eta = 'ak5PFJets_eta_'
    phi = 'ak5PFJets_phi_'
    mass = 'ak5PFJets_mass_'
    test_data = test_data[[pt, eta, phi, mass]]
    #predicted_data = predicted_data[[pt, eta, phi, mass]]

    figure = corner.corner(test_data)
    #corner.corner(predicted_data, fig=figure, color='red')
    plt.show()
