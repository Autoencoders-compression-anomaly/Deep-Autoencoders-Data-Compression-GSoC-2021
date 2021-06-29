import matplotlib.pyplot as plt
import pandas as pd


def plot():
    data_df = pd.read_csv('27D_openCMS_preprocessed_data.csv')

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

    prefix = 'ak5PFJets.'

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

    n_bins = 100

    for kk in range(0, 28):
        if branches[kk] == prefix + 'pt_' or branches[kk] == prefix + 'mass_':
            n_hist_data, bin_edges, _ = plt.hist(data_df[branches[kk]], color='orange', label='Input', alpha=1, bins=n_bins, log=True)
            plt.xlabel(xlabel=variable_list[kk])
            plt.ylabel('# of events')
        else:
            n_hist_data, bin_edges, _ = plt.hist(data_df[branches[kk]], color='orange', label='Input', alpha=1, bins=n_bins)
            plt.xlabel(xlabel=variable_list[kk])
            plt.ylabel('# of events')
        plt.show()

plot()