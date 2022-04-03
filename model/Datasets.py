import torch
import numpy as np
from utils import data_preprocess
import pickle


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
    ali_num_feat_dim = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

    if pars.dataset == 'tiny-criteo':
        field_size = 39
        train_dict = data_preprocess.read_data('./data/tiny_train_input.csv', './data/category_emb',
                                               criteo_num_feat_dim,
                                               feature_dim_start=0, dim=field_size)
        valid_dict = data_preprocess.read_data('./data/tiny_test_input.csv', './data/category_emb', criteo_num_feat_dim,
                                               feature_dim_start=0, dim=field_size)
        test_dict = data_preprocess.read_data('./data/tiny_test_input.csv', './data/category_emb', criteo_num_feat_dim,
                                              feature_dim_start=0, dim=field_size)

    elif pars.dataset == 'twitter':
        field_size = 47
        pars.numerical = 11
        train_dict = data_preprocess.read_data_twitter('./data/large/twitter_train_s.parquet',
                                                       './data/large/twitter_feature_map_s',
                                                       twitter_num_feat_dim, feature_dim_start=4, dim=field_size,
                                                       twitter_category=pars.twitter_category)
        valid_dict = data_preprocess.read_data_twitter('./data/large/twitter_valid_s.parquet',
                                                       './data/large/twitter_feature_map_s',
                                                       twitter_num_feat_dim, feature_dim_start=4, dim=field_size,
                                                       twitter_category=pars.twitter_category)
        test_dict = data_preprocess.read_data_twitter('./data/large/twitter_test_s.parquet',
                                                      './data/large/twitter_feature_map_s',
                                                      twitter_num_feat_dim, feature_dim_start=4, dim=field_size,
                                                      twitter_category=pars.twitter_category)

    elif pars.dataset == 'ali':
        field_size = 20
        train_dict = data_preprocess.read_data_ali('./data/large/ali_train.csv', './data/large/ali_feature_map',
                                                   ali_num_feat_dim, feature_dim_start=1, dim=20)
        valid_dict = data_preprocess.read_data_ali('./data/large/ali_valid.csv', './data/large/ali_feature_map',
                                                   ali_num_feat_dim, feature_dim_start=1, dim=20)
        test_dict = data_preprocess.read_data_ali('./data/large/ali_test.csv', './data/large/ali_feature_map',
                                                  ali_num_feat_dim, feature_dim_start=1, dim=20)


    else:  # criteo dataset
        field_size = 39
        train_dict = data_preprocess.read_data('./data/large/criteo_train.csv', './data/large/criteo_feature_map',
                                               criteo_num_feat_dim, feature_dim_start=1, dim=39)
        valid_dict = data_preprocess.read_data('./data/large/criteo_valid.csv', './data/large/criteo_feature_map',
                                               criteo_num_feat_dim, feature_dim_start=1, dim=39)
        test_dict = data_preprocess.read_data('./data/large/criteo_test.csv', './data/large/criteo_feature_map',
                                              criteo_num_feat_dim, feature_dim_start=1, dim=39)

    return field_size, train_dict, valid_dict, test_dict
