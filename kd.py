import random
import argparse
import sys

import numpy as np

from model import DeepFMs
from utils import data_preprocess
from utils.parameters import get_parser
from utils.util import get_model, load_model_dic, get_logger

from model.Datasets import Dataset, get_dataset

import torch
import warnings

"""
source: https://github.com/peterliht/knowledge-distillation-pytorch
"""
#warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = get_parser()
    pars = parser.parse_args()

    logger = get_logger('Knowledge Distillation')
    logger.info(pars)

    if not pars.save_model_path:
        logger.error("no model path given: -save_model_path")
        sys.exit()

    field_size, train_dict, valid_dict, test_dict = get_dataset(pars)

    # teacher
    model = get_model(field_size=field_size, cuda=pars.use_cuda and torch.cuda.is_available(), feature_sizes=train_dict['feature_sizes'], pars=pars, logger=logger)
    model = load_model_dic(model, pars.save_model_path)

    # student
    number_of_deep_nodes = 32
    h_depth = 1
    student = get_model(field_size=field_size, cuda=pars.use_cuda and torch.cuda.is_available(), feature_sizes=train_dict['feature_sizes'], deep_nodes=number_of_deep_nodes, h_depth=h_depth,
                        pars=pars, logger=logger)

    logger.info(model)
    logger.info(student)

    if pars.use_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        student = student.cuda()
        model = model.cuda()

    logger.info("Train student model")
    student.fit(train_dict['index'], train_dict['value'], train_dict['label'], valid_dict['index'],
                valid_dict['value'], valid_dict['label'],
                prune=pars.prune, prune_fm=pars.prune_fm, prune_r=pars.prune_r, prune_deep=pars.prune_deep,
                save_path=pars.save_model_path + '_kd', emb_r=pars.emb_r, emb_corr=pars.emb_corr, teacher_model=model)


    logger.info('Original model:')
    model = get_model(field_size=field_size, cuda=0, feature_sizes=train_dict['feature_sizes'], pars=pars, logger=logger)
    model = load_model_dic(model, pars.save_model_path)
    f = model.print_size_of_model()
    model.run_benchmark(test_dict['index'], test_dict['value'], test_dict['label'])

    logger.info('Student model:')
    student = get_model(field_size=field_size, cuda=0, feature_sizes=train_dict['feature_sizes'], deep_nodes=number_of_deep_nodes, h_depth=h_depth,
                        pars=pars, logger=logger)
    student = load_model_dic(student, pars.save_model_path + '_kd')
    s = student.print_size_of_model()
    logger.info("\t{0:.2f} times smaller".format(f / s))
    student.run_benchmark(test_dict['index'], test_dict['value'], test_dict['label'])