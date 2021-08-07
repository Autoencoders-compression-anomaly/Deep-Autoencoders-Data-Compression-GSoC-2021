import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

"""
This file includes functions for preprocessing the initial data
"""


def preprocess_28D(data_df, num_variables, min_max_all):
    data_df = data_df.drop(['entry', 'subentry'], axis=1)
    data_df = data_df.sort_values(by=['ak5PFJets_pt_'])

    # drop variables that have only zero values
    data_df = data_df.drop(['ak5PFJets_fX', 'ak5PFJets_fY', 'ak5PFJets_fZ', 'ak5PFJets_mPileupEnergy'], axis=1)

    if num_variables == 19:
        data_df = data_df.drop(['ak5PFJets_mChargedEmEnergy', 'ak5PFJets_mChargedMuEnergy', 'ak5PFJets_mMuonEnergy',
                                'ak5PFJets_mMuonMultiplicity', 'ak5PFJets_mElectronEnergy'], axis=1)

    # filter out jets having pT > 8 TeV
    data_df = data_df[data_df.ak5PFJets_pt_ < 8000]
    data_df = data_df[data_df.ak5PFJets_mass_ < 800]

    # Normalize our data using Standard Scaler from sklearn
    # scaler = StandardScaler()
    # data_df[data_df.columns] = scaler.fit_transform(data_df)
    min_max_scaler = MinMaxScaler()

    if min_max_all:
        # Normalize all variables in the range (0, 1) using MinMax Scaler from sklearn
        data_df[data_df.columns] = min_max_scaler.fit_transform(data_df)
    else:
        # Perform a custom normalization technique for the 4-momentum variables
        temp_df = data_df.copy()
        temp_df = custom_normalization(temp_df)

        # Normalize the rest of the variables in the range (0, 1) using MinMax Scaler from sklearn
        data_df[data_df.columns] = min_max_scaler.fit_transform(data_df)

        data_df['ak5PFJets_pt_'] = temp_df['ak5PFJets_pt_'].values
        data_df['ak5PFJets_phi_'] = temp_df['ak5PFJets_phi_'].values
        data_df['ak5PFJets_eta_'] = temp_df['ak5PFJets_eta_'].values
        data_df['ak5PFJets_mass_'] = temp_df['ak5PFJets_mass_'].values

    print('Normalized data:')
    print(data_df)
    # shuffling the data before splitting
    data_df = shuffle(data_df)

    # split the data into train and test with a ratio of 15%
    train_set, test_set = train_test_split(data_df, test_size=0.15, random_state=1)

    print('Train data shape: ')
    print(train_set.shape)
    print('Test data shape: ')
    print(test_set.shape)

    # save preprocessed data to a csv file
    data_df.to_csv('28D_openCMS_preprocessed_data.csv')

    return data_df, train_set, test_set, min_max_scaler


"""
Custom normalization for the 4-momentum variables
"""
def custom_normalization(data):
    pt_div = 1.2
    pt_sub = 1.3
    phi_div = 3
    eta_div = 5
    m_div = 1.8
    m_add = 1

    data['ak5PFJets_eta_'] = data['ak5PFJets_eta_'] / eta_div
    data['ak5PFJets_phi_'] = data['ak5PFJets_phi_'] / phi_div
    data['ak5PFJets_mass_'] = np.log10(data['ak5PFJets_mass_'] + m_add) / m_div
    data['ak5PFJets_pt_'] = (np.log10(data['ak5PFJets_pt_']) - pt_sub) / pt_div

    return data


def preprocess_4D(input_path):
    raw_data = []
    print('Reading data at: ', input_path)
    with open(input_path, 'r') as file:
        for line in file.readlines():
            line = line.replace(';', ',')
            line = line.rstrip(',\n')
            line = line.split(',')
            raw_data.append(line)

    # Find the longest line in the data
    longest_line = max(raw_data, key=len)

    # Set the maximum number of columns
    max_col_num = len(longest_line)

    # Set the columns names
    col_names = ['event_ID', 'process_ID', 'event_weight', 'MET', 'MET_Phi']
    meta_cols = col_names.copy()

    # Create columns names for "four-momentum" of the jet particles
    for i in range(1, (int((max_col_num - 5) / 5)) + 1):
        col_names.append('obj' + str(i))
        col_names.append('E' + str(i))
        col_names.append('pt' + str(i))
        col_names.append('eta' + str(i))
        col_names.append('phi' + str(i))

    # Create a pandas dataframe to store the whole data
    df = pd.DataFrame(raw_data, columns=col_names)
    df.fillna(value=np.nan, inplace=True)

    # Create a pandas dataframe to store only the data we need for training our AE (i.e. "four-momentum" of the jet particles)
    data = pd.DataFrame(df.values, columns=col_names)
    data.fillna(value=0, inplace=True)
    # Drop unnecessary columns
    data.drop(columns=meta_cols, inplace=True)

    ignore_particles = ['e-', 'e+', 'm-', 'm+', 'g', 'b']
    ignore_list = []
    for i in range(len(data)):
        for j in data.loc[i].keys():
            if 'obj' in j:
                if data.loc[i][j] in ignore_particles:
                    ignore_list.append(i)
                    break

    data.drop(ignore_list, inplace=True)

    x = data.values.reshape([data.shape[0] * data.shape[1] // 5, 5])

    temp_list = []
    for i in range(x.shape[0]):
        if (x[i] == 0).all():
            temp_list.append(i)
    x1 = np.delete(x, temp_list, 0)
    del x

    temp_list = []
    for i in range(x1.shape[0]):
        if (x1[i][0] == 'j'):
            continue
        else:
            temp_list.append(i)
            print(i, x1[i][0])

    data = np.delete(x1, temp_list, 0)

    col_names = ['obj', 'E', 'pt', 'eta', 'phi']
    data_df = pd.DataFrame(data, columns=col_names)
    # Drop the 'obj' column as it's unnecessary
    data_df.drop(columns='obj', inplace=True)
    data_df = data_df.astype('float32')

    print('Data after preprocessing: ')
    print(data_df)
    # Standardize data Standard Scalar from sklearn
    data_df[data_df.columns] = StandardScaler().fit_transform(data_df)
    print('Normalized data:')
    print(data_df)
    # shuffling the data before splitting
    data_df = shuffle(data_df)

    # split the data into train and test with a ratio of 20%
    train_set, test_set = train_test_split(data_df, test_size=0.2, random_state=1)
    return data_df, train_set, test_set


"""
    A function to scale back the data to the original representation and 
    handle the integer-variables which have been transformed to floats during normalization
"""


def post_process_28D(data_df, scaler):
    # TODO
    data_df[data_df.columns] = scaler.inverse_transform(data_df)

    return data_df
