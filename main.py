from data_preprocessing import preprocess
from autoencoders import standard_AE


train_data, test_data = preprocess()

standard_AE.train(train_data=train_data, test_data=test_data)