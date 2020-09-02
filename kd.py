import random
import argparse
import sys

import numpy as np

from model import DeepFMs
from utils import data_preprocess
from utils.parameters import getParser
from utils.util import get_model, load_model

import torch
import warnings

"""
source: https://github.com/peterliht/knowledge-distillation-pytorch
"""
warnings.filterwarnings("ignore")

parser = getParser()
pars = parser.parse_args()

criteo_num_feat_dim = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
field_size = 39
train_dict = data_preprocess.read_data('./data/large/train_criteo_s.csv', './data/large/criteo_feature_map_s',
                                       criteo_num_feat_dim, feature_dim_start=1, dim=39)
valid_dict = data_preprocess.read_data('./data/large/valid_criteo_s.csv', './data/large/criteo_feature_map_s',
                                       criteo_num_feat_dim, feature_dim_start=1, dim=39)

if not pars.save_model_name:
    print("no model path given: -save_model_name")
    sys.exit()

if __name__ == '__main__':
    model = get_model(cuda=1, feature_sizes=train_dict['feature_sizes'], pars=pars)
    model = load_model(model, pars.save_model_name)

    number_of_deep_nodes = 50

    student = get_model(cuda=pars.use_cuda and torch.cuda.is_available(), feature_sizes=train_dict['feature_sizes'], deep_nodes=number_of_deep_nodes,
                        pars=pars)

    if pars.use_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        student = student.cuda()
        model = model.cuda()

    student.fit(train_dict['index'], train_dict['value'], train_dict['label'], valid_dict['index'],
                valid_dict['value'], valid_dict['label'],
                prune=pars.prune, prune_fm=pars.prune_fm, prune_r=pars.prune_r, prune_deep=pars.prune_deep,
                save_path=pars.save_model_name + '_kd', emb_r=pars.emb_r, emb_corr=pars.emb_corr, teacher_model=model)


    print('Original model:')
    model = get_model(cuda=0, feature_sizes=train_dict['feature_sizes'], pars=pars)
    model = load_model(model, pars.save_model_name)
    f = model.print_size_of_model()
    model.time_model_evaluation(valid_dict['index'], valid_dict['value'], valid_dict['label'])

    print('Student model:')
    student = get_model(cuda=0, feature_sizes=train_dict['feature_sizes'], deep_nodes=number_of_deep_nodes, pars=pars)
    student = load_model(student, pars.save_model_name + '_kd')
    s = student.print_size_of_model()
    print("\t{0:.2f} times smaller".format(f / s))
    student.time_model_evaluation(valid_dict['index'], valid_dict['value'], valid_dict['label'])