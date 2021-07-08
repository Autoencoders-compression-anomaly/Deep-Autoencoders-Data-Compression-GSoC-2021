import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_initial_data(input_data):
    save_dir = "D:\Desktop\GSoC-ATLAS\initial_data_plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    variable_list = ['pt_', 'eta_', 'phi_', 'mass_',
                     'fX', 'fY', 'fZ', 'mJetArea',
                     'mPileupEnergy', 'mChargedHadronEnergy', 'mNeutralHadronEnergy',
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
                prefix + 'fX', prefix + 'fY', prefix + 'fZ', prefix + 'mJetArea',
                prefix + 'mPileupEnergy', prefix + 'mChargedHadronEnergy', prefix + 'mNeutralHadronEnergy',
                prefix + 'mPhotonEnergy',
                prefix + 'mElectronEnergy', prefix + 'mMuonEnergy', prefix + 'mHFHadronEnergy',
                prefix + 'mHFEMEnergy', prefix + 'mChargedHadronMultiplicity',
                prefix + 'mNeutralHadronMultiplicity',
                prefix + 'mPhotonMultiplicity', prefix + 'mElectronMultiplicity',
                prefix + 'mMuonMultiplicity',
                prefix + 'mHFHadronMultiplicity', prefix + 'mHFEMMultiplicity', prefix + 'mChargedEmEnergy',
                prefix + 'mChargedMuEnergy', prefix + 'mNeutralEmEnergy', prefix + 'mChargedMultiplicity',
                prefix + 'mNeutralMultiplicity']

    for kk in range(0, 28):
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


def plot_test_pred_data(test_data, predicted_data):
    save_dir = "D:\Desktop\GSoC-ATLAS\AE_plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    variable_list = ['pt_', 'eta_', 'phi_', 'mass_',
                     'fX', 'fY', 'fZ', 'mJetArea',
                     'mPileupEnergy', 'mChargedHadronEnergy', 'mNeutralHadronEnergy',
                     'mPhotonEnergy',
                     'mElectronEnergy', 'mMuonEnergy', 'mHFHadronEnergy',
                     'mHFEMEnergy', 'mChargedHadronMultiplicity',
                     'mNeutralHadronMultiplicity',
                     'mPhotonMultiplicity', 'mElectronMultiplicity',
                     'mMuonMultiplicity',
                     'mHFHadronMultiplicity', 'mHFEMMultiplicity', 'mChargedEmEnergy',
                     'mChargedMuEnergy', 'mNeutralEmEnergy', 'mChargedMultiplicity',
                     'mNeutralMultiplicity']

    colors = ['pink', 'green']
    prefix = 'ak5PFJets_'
    n_bins = 100
    save = True  # Option to save figure

    #predicted_data = predicted_data.detach().numpy()
    test_data = test_data.values

    # plot the input data along with the reconstructed from the AE
    for kk in np.arange(28):
        plt.figure()
        n_hist_data, bin_edges, _ = plt.hist(test_data[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
        n_hist_pred, _, _ = plt.hist(predicted_data[:, kk], color=colors[0], label='Output', alpha=0.8, bins=bin_edges)
        plt.suptitle(variable_list[kk])
        plt.xlabel(xlabel=variable_list[kk])
        plt.ylabel('Number of jets')
        # ms.sciy()
        plt.yscale('log')
        plt.legend()
        if save:
            plt.savefig(os.path.join(save_dir, variable_list[kk] + '.png'))


# plot(data_df = pd.read_csv('27D_openCMS_preprocessed_data.csv'))
