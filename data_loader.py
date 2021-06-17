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
    recoPFJets.show(name_width=100, typename_width=100)

    prefix = 'recoPFJets_ak5PFJets__RECO.obj.'
    # Store the data in a pandas dataframe
    dataframe = recoPFJets.arrays(
        [prefix + 'qx3_', prefix + 'pt_', prefix + 'eta_', prefix + 'phi_', prefix + 'mass_'],
        library="pd")

    dataframe.columns = ['qx3_', 'pt_', 'eta_', 'phi_', 'mass_']

    print("\nDataframe:")
    print(dataframe.head())

    return dataframe


load_cms_data()