from data_preprocessing import preprocess
from autoencoders import standard_AE
from data_loader import load_cms_data

#cms_data_df = load_cms_data(filename="open_cms_data.root")
train_data, test_data = preprocess()

standard_AE.train(train_data=train_data, test_data=test_data)