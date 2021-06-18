import uproot
import pandas as pd
import numpy as np


def load_cms_data(filename="open_cms_data.root"):
    """This function loads events data from open CMS root files"""

    # The object returned by uproot.open represents a TDirectory inside the file (/).
    # We are interested in the Events branch
    events_tree = uproot.open(filename)['Events']

    # events_tree.show(name_width=100, typename_width=100)

    # The Collection we want is: recoPFJets_ak5PFJets__RECO

    recoPFJets = events_tree['recoPFJets_ak5PFJets__RECO.']['recoPFJets_ak5PFJets__RECO.obj']
    #recoPFJets.show(name_width=100, typename_width=100)

    prefix = 'recoPFJets_ak5PFJets__RECO.obj.'
    # Store the 27 variables we are interested in to a pandas dataframe
    dataframe = recoPFJets.arrays(
        [prefix + 'pt_', prefix + 'eta_', prefix + 'phi_', prefix + 'mass_', prefix + 'vertex_.fCoordinates.fX',
         prefix + 'vertex_.fCoordinates.fY', prefix + 'vertex_.fCoordinates.fZ', prefix + 'mJetArea', prefix + 'mPileupEnergy',
         prefix + 'm_specific.mChargedHadronEnergy', prefix + 'm_specific.mNeutralHadronEnergy',
         prefix + 'm_specific.mPhotonEnergy', prefix + 'm_specific.mElectronEnergy',
         prefix + 'm_specific.mMuonEnergy', prefix + 'm_specific.mHFHadronEnergy', prefix + 'm_specific.mHFEMEnergy',
         prefix + 'm_specific.mChargedHadronMultiplicity', prefix + 'm_specific.mNeutralHadronMultiplicity',
         prefix + 'm_specific.mPhotonMultiplicity', prefix + 'm_specific.mElectronMultiplicity', prefix + 'm_specific.mMuonMultiplicity',
         prefix + 'm_specific.mHFHadronMultiplicity', prefix + 'm_specific.mHFEMMultiplicity',
         prefix + 'm_specific.mChargedEmEnergy', prefix + 'm_specific.mChargedMuEnergy', prefix + 'm_specific.mNeutralEmEnergy',
         prefix + 'm_specific.mChargedMultiplicity', prefix + 'm_specific.mNeutralMultiplicity'],       library="pd")

    prefix2 = 'ak5PFJets.'
    # Rename the column names to be shorter
    dataframe.columns = [prefix2 + 'pt_', prefix2 + 'eta_', prefix2 + 'phi_', prefix2 + 'mass_',
                         prefix2 + 'fX', prefix2 + 'fY', prefix2 + 'fZ', prefix2 + 'mJetArea', prefix2 + 'mPileupEnergy',
                         prefix2 + 'mChargedHadronEnergy', prefix2 + 'mNeutralHadronEnergy', prefix2 + 'mPhotonEnergy',
                         prefix2 + 'mElectronEnergy', prefix2 + 'mMuonEnergy', prefix2 + 'mHFHadronEnergy',
                         prefix2 + 'mHFEMEnergy', prefix2 + 'mChargedHadronMultiplicity', prefix2 + 'mNeutralHadronMultiplicity',
                         prefix2 + 'mPhotonMultiplicity', prefix2 + 'mElectronMultiplicity', prefix2 + 'mMuonMultiplicity',
                         prefix2 + 'mHFHadronMultiplicity', prefix2 + 'mHFEMMultiplicity', prefix2 + 'mChargedEmEnergy',
                         prefix2 + 'mChargedMuEnergy', prefix2 + 'mNeutralEmEnergy', prefix2 + 'mChargedMultiplicity',
                         prefix2 + 'mNeutralMultiplicity']


    print("\nDataframe:")
    print(dataframe.head())
    dataframe.to_csv('27D_opensCMS_data.csv')
    return dataframe


load_cms_data()