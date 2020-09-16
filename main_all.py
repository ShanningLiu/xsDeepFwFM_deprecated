import random
import numpy as np
import torch
from distiller.quantization import PostTrainLinearQuantizer, LinearQuantMode
from copy import deepcopy

from model import DeepFMs
from model.Datasets import Dataset
from utils import data_preprocess
from utils.parameters import getParser
from utils.util import get_model, load_model

parser = getParser()
pars = parser.parse_args()

if __name__ == '__main__':
    print(pars)
    np.random.seed(pars.random_seed)
    random.seed(pars.random_seed)
    torch.manual_seed(pars.random_seed)
    torch.cuda.manual_seed(pars.random_seed)

    save_model_name = './saved_models/' + pars.c + '_l2_' + str(pars.l2) + '_sparse_' + str(
        pars.sparse) + '_seed_' + str(pars.random_seed)

    if pars.qr_flag:
        save_model_name = save_model_name + '_qr'
    if pars.md_flag:
        save_model_name = save_model_name + '_md'

    criteo_num_feat_dim = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    twitter_num_feat_dim = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    if pars.dataset == 'tiny-criteo':
        field_size = 39
        train_dict = data_preprocess.read_data('./data/tiny_train_input.csv', './data/category_emb',
                                               criteo_num_feat_dim,
                                               feature_dim_start=0, dim=39)
        valid_dict = data_preprocess.read_data('./data/tiny_test_input.csv', './data/category_emb', criteo_num_feat_dim,
                                               feature_dim_start=0, dim=39)
    elif pars.dataset == 'twitter':
        field_size = 20
        pars.numerical = 15
        train_dict = data_preprocess.read_data('./data/large/train_twitter_s.csv', './data/large/twitter_feature_map_s',
                                               twitter_num_feat_dim, feature_dim_start=1, dim=20)
        valid_dict = data_preprocess.read_data('./data/large/valid_twitter_s.csv', './data/large/twitter_feature_map_s',
                                               twitter_num_feat_dim, feature_dim_start=1, dim=20)
    else:  # criteo dataset
        field_size = 39
        train_dict = data_preprocess.read_data('./data/large/train_criteo_s.csv', './data/large/criteo_feature_map_s',
                                               criteo_num_feat_dim, feature_dim_start=1, dim=39)
        valid_dict = data_preprocess.read_data('./data/large/valid_criteo_s.csv', './data/large/criteo_feature_map_s',
                                               criteo_num_feat_dim, feature_dim_start=1, dim=39)

    model = get_model(cuda=pars.use_cuda and torch.cuda.is_available(), feature_sizes=train_dict['feature_sizes'],
                      pars=pars)
    if pars.use_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        model = model.cuda()

    print(model)

    model.fit(train_dict['index'], train_dict['value'], train_dict['label'], valid_dict['index'],
              valid_dict['value'], valid_dict['label'],
              prune=pars.prune, prune_fm=pars.prune_fm, prune_r=pars.prune_r, prune_deep=pars.prune_deep,
              save_path=save_model_name, emb_r=pars.emb_r, emb_corr=pars.emb_corr)

    # measurement
    time_on_cuda = False
    model = get_model(cuda=time_on_cuda, feature_sizes=train_dict['feature_sizes'], pars=pars)
    model = load_model(model, save_model_name)
    if time_on_cuda:
        model = model.cuda()

    model.print_size_of_model()
    model.time_model_evaluation(valid_dict['index'], valid_dict['value'], valid_dict['label'], cuda=time_on_cuda)
