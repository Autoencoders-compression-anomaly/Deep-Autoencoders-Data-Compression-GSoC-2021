import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def preprocess(filename='27D_opensCMS_data.csv'):
    data_df = pd.read_csv(filename)
    data_df = data_df.drop(['entry', 'subentry'], axis=1)
    # data_df = data_df.astype('float32')

    # Standardize our data using Standard Scalar from sklearn
    data_df[data_df.columns] = StandardScaler().fit_transform(data_df)
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

    data_df.to_csv('27D_opensCMS_preprocessed_data.csv')

    return train_set, test_set


preprocess()