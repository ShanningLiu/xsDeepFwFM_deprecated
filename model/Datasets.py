import torch
import numpy as np
from utils import data_preprocess

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X_index, X_value, labels, size):
        self.labels = labels
        self.X_index = np.array(X_index).reshape((-1, size, 1))
        self.X_value = np.array(X_value)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        Xi_i = torch.tensor(self.X_index[index]).long()
        Xv_i = torch.tensor(self.X_value[index]).long()
        y = torch.tensor(self.labels[index]).float()

        return Xi_i, Xv_i, y


def get_dataset(pars):
    criteo_num_feat_dim = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    twitter_num_feat_dim = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    if pars.dataset == 'tiny-criteo':
        field_size = 39
        train_dict = data_preprocess.read_data('./data/tiny_train_input.csv', './data/category_emb',
                                               criteo_num_feat_dim,
                                               feature_dim_start=0, dim=field_size)
        valid_dict = data_preprocess.read_data('./data/tiny_test_input.csv', './data/category_emb', criteo_num_feat_dim,
                                               feature_dim_start=0, dim=field_size)

    elif pars.dataset == 'twitter':
        field_size = 47
        pars.numerical = 11
        train_dict = data_preprocess.read_data('./data/large/twitter_train.parquet', './data/large/twitter_feature_map',
                                               twitter_num_feat_dim, feature_dim_start=1, dim=field_size, parquet=True)
        valid_dict = data_preprocess.read_data('./data/large/twitter_valid.parquet', './data/large/twitter_feature_map',
                                               twitter_num_feat_dim, feature_dim_start=1, dim=field_size, parquet=True)

    else:  # criteo dataset
        field_size = 39
        '''train_dict = data_preprocess.read_data('./data/large/train_criteo_s.csv', './data/large/criteo_feature_map_s',
                                               criteo_num_feat_dim, feature_dim_start=1, dim=39)
        valid_dict = data_preprocess.read_data('./data/large/valid_criteo_s.csv', './data/large/criteo_feature_map_s',
                                               criteo_num_feat_dim, feature_dim_start=1, dim=39)'''
        train_dict = data_preprocess.get_feature_sizes('./data/large/full_criteo_feature_map',
                                                       criteo_num_feat_dim, feature_dim_start=1, dim=39)
        valid_dict = data_preprocess.read_data('./data/large/full_valid_criteo.csv',
                                               './data/large/full_criteo_feature_map',
                                               criteo_num_feat_dim, feature_dim_start=1, dim=39)

    return field_size, train_dict, valid_dict