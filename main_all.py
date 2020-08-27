#!/usr/bin/env python

import random
import argparse

import numpy as np

from model import DeepFMs
from model.Datasets import Dataset
from utils import data_preprocess
from model.Huffmancoding import huffman_encode_model
from model.WeightSharing import apply_weight_sharing

import torch

parser = argparse.ArgumentParser(description='Hyperparameter tuning')
parser.add_argument('-c', default='DeepFwFM', type=str, help='Models: FM, DeepFwFM ...')
parser.add_argument('-use_cuda', default=1, type=int, help='Use CUDA or not')
parser.add_argument('-gpu', default=0, type=int, help='GPU id')
parser.add_argument('-n_epochs', default=8, type=int, help='Number of epochs')
parser.add_argument('-numerical', default=13, type=int, help='Numerical features, 13 for Criteo')
parser.add_argument('-use_multi', default='0', type=int, help='Use multiple CUDAs')
parser.add_argument('-use_logit', default=0, type=int, help='Use Logistic regression')
parser.add_argument('-use_fm', default=0, type=int, help='Use FM module or not')
parser.add_argument('-use_fwlw', default=0, type=int, help='If to include FwFM linear weights or not')
parser.add_argument('-use_lw', default=1, type=int, help='If to include FM linear weights or not')
parser.add_argument('-use_ffm', default=0, type=int, help='Use FFM module or not')
parser.add_argument('-use_fwfm', default=1, type=int, help='Use FwFM module or not')
parser.add_argument('-use_deep', default=1, type=int, help='Use Deep module or not')
parser.add_argument('-num_deeps', default=1, type=int, help='Number of deep networks')
parser.add_argument('-deep_nodes', default=400, type=int, help='Nodes in each layer')
parser.add_argument('-h_depth', default=3, type=int, help='Deep layers')
parser.add_argument('-prune', default=0, type=int, help='Prune model or not')
parser.add_argument('-prune_r', default=0, type=int, help='Prune r')
parser.add_argument('-prune_deep', default=1, type=int, help='Prune Deep component')
parser.add_argument('-prune_fm', default=1, type=int, help='Prune FM component')
parser.add_argument('-emb_r', default=1., type=float, help='Sparse FM ratio over Sparse Deep ratio')
parser.add_argument('-emb_corr', default=1., type=float, help='Sparse Corr ratio over Sparse Deep ratio')
parser.add_argument('-sparse', default=0.9, type=float, help='Sparse rate')
parser.add_argument('-warm', default=10, type=float, help='Warm up epochs before pruning')
parser.add_argument('-ensemble', default=0, type=int, help='Ensemble models or not')
parser.add_argument('-embedding_size', default=10, type=int, help='Embedding size')
parser.add_argument('-batch_size', default=4096, type=int, help='Batch size')
parser.add_argument('-random_seed', default=42, type=int, help='Random seed')
parser.add_argument('-learning_rate', default=0.001, type=float, help='Learning rate')
parser.add_argument('-momentum', default=0, type=float, help='Momentum')
parser.add_argument('-l2', default=3e-7, type=float, help='L2 penalty')
parser.add_argument('-dataset', default='criteo', type=str, help='Dataset to use')
parser.add_argument('-generator', default=1, type=int, help='Use generator')
parser.add_argument('-dynamic_quantization', default=0, type=int, help='Apply dynamic network quantization')
parser.add_argument('-static_quantization', default=0, type=int, help='Apply static network quantization')
parser.add_argument('-quantization_aware', default=0, type=int, help='Quantization Aware Training')
parser.add_argument('-weight_sharing', default=0, type=int, help='Apply K-means clustering algorithm for weights')
parser.add_argument('-huffman_encoding', default=0, type=int, help='Apply Huffman coding algorithm for each of the weights in the network')
pars = parser.parse_args()


def load_model(model, model_file):
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    #model.to('cpu')
    return model


def get_model(cuda=1, quantization_aware=0, dynamic_quantization=0,
                           static_quantization=0):
    return DeepFMs.DeepFMs(field_size=field_size, feature_sizes=train_dict['feature_sizes'],
                           embedding_size=pars.embedding_size, n_epochs=pars.n_epochs,
                           verbose=False, use_cuda=cuda, use_fm=pars.use_fm, use_fwfm=pars.use_fwfm,
                           use_ffm=pars.use_ffm, use_deep=pars.use_deep,
                           batch_size=pars.batch_size, learning_rate=pars.learning_rate, weight_decay=pars.l2,
                           momentum=pars.momentum, sparse=pars.sparse, warm=pars.warm,
                           h_depth=pars.h_depth, deep_nodes=pars.deep_nodes, num_deeps=pars.num_deeps,
                           numerical=pars.numerical, use_lw=pars.use_lw, use_fwlw=pars.use_fwlw,
                           use_logit=pars.use_logit, random_seed=pars.random_seed,
                           quantization_aware=quantization_aware, dynamic_quantization=dynamic_quantization,
                           static_quantization=static_quantization)


if __name__ == '__main__':
    print(pars)
    np.random.seed(pars.random_seed)
    random.seed(pars.random_seed)
    torch.manual_seed(pars.random_seed)
    torch.cuda.manual_seed(pars.random_seed)

    save_model_name = './saved_models/' + pars.c + '_l2_' + str(pars.l2) + '_sparse_' + str(
        pars.sparse) + '_seed_' + str(
        pars.random_seed)

    criteo_num_feat_dim = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    twitter_num_feat_dim = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    if pars.dataset == 'tiny-criteo':
        field_size = 39
        index_size = 26
        train_dict = data_preprocess.read_data('./data/tiny_train_input.csv', './data/category_emb',
                                               criteo_num_feat_dim,
                                               feature_dim_start=0, dim=39)
        valid_dict = data_preprocess.read_data('./data/tiny_test_input.csv', './data/category_emb', criteo_num_feat_dim,
                                               feature_dim_start=0, dim=39)
    elif pars.dataset == 'twitter':
        field_size = 20
        pars.numerical = 15
        index_size = 5
        train_dict = data_preprocess.read_data('./data/large/train_twitter_s.csv', './data/large/twitter_feature_map_s',
                                               twitter_num_feat_dim, feature_dim_start=1, dim=20)
        valid_dict = data_preprocess.read_data('./data/large/valid_twitter_s.csv', './data/large/twitter_feature_map_s',
                                               twitter_num_feat_dim, feature_dim_start=1, dim=20)
    else:  # criteo dataset
        field_size = 39
        index_size = 26
        train_dict = data_preprocess.read_data('./data/large/train_criteo_s.csv', './data/large/criteo_feature_map_s',
                                               criteo_num_feat_dim, feature_dim_start=1, dim=39)
        valid_dict = data_preprocess.read_data('./data/large/valid_criteo_s.csv', './data/large/criteo_feature_map_s',
                                               criteo_num_feat_dim, feature_dim_start=1, dim=39)

    model = get_model(cuda=pars.use_cuda)
    if pars.use_cuda:
        model = model.cuda()
    if pars.quantization_aware:
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        # print(model.qconfig)
        torch.quantization.prepare(model, inplace=True)

    if pars.generator == 1:
        params = {'batch_size': pars.batch_size,
                  'shuffle': True,
                  'num_workers': 0}

        training_set = Dataset(train_dict['index'], train_dict['value'], train_dict['label'], size=index_size)
        training_generator = torch.utils.data.DataLoader(training_set, **params)

        valid_set = Dataset(valid_dict['index'], valid_dict['value'], valid_dict['label'], size=index_size)
        valid_generator = torch.utils.data.DataLoader(valid_set, **params)
        model.fit_generator(training_generator, valid_generator,
                            prune=pars.prune, prune_fm=pars.prune_fm, prune_r=pars.prune_r, prune_deep=pars.prune_deep,
                            save_path=save_model_name, emb_r=pars.emb_r, emb_corr=pars.emb_corr)

    else:
        model.fit(train_dict['index'], train_dict['value'], train_dict['label'], valid_dict['index'],
                  valid_dict['value'], valid_dict['label'],
                  prune=pars.prune, prune_fm=pars.prune_fm, prune_r=pars.prune_r, prune_deep=pars.prune_deep,
                  save_path=save_model_name, emb_r=pars.emb_r, emb_corr=pars.emb_corr)

    model = load_model(get_model(cuda=0), save_model_name) # no cuda
    model.print_size_of_model()
    model.time_model_evaluation(valid_dict['index'], valid_dict['value'], valid_dict['label'])

    """
    TODO (self.field_cov.weight.t() + self.field_cov.weight) * 0.5) # TODO weight tensor from quantized linear
    AttributeError: 'function' object has no attribute 't'
    """
    # quantization (no CUDA allowed and dynamic after training)
    if pars.dynamic_quantization:
        quantized_model = load_model(get_model(cuda=0, dynamic_quantization=1), save_model_name)
        print(quantized_model)
        quantized_model.eval()

        quantized_model = torch.quantization.quantize_dynamic(quantized_model, {torch.nn.Linear}, dtype=torch.qint8)
        print(quantized_model)
        quantized_model.print_size_of_model()
        quantized_model.time_model_evaluation(valid_dict['index'], valid_dict['value'], valid_dict['label'])

    if pars.static_quantization: # https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
        quantized_model = load_model(get_model(cuda=0, static_quantization=1), save_model_name)
        quantized_model.eval()

        quantized_model.qconfig = torch.quantization.default_qconfig
        print(quantized_model.qconfig)
        torch.quantization.prepare(quantized_model, inplace=True)

        # Calibrate
        quantized_model.time_model_evaluation(train_dict['index'], train_dict['value'], train_dict['label'])
        print('Post Training Quantization: Calibration done')

        # Convert to quantized model
        torch.quantization.convert(quantized_model, inplace=True)
        quantized_model.time_model_evaluation(valid_dict['index'], valid_dict['value'], valid_dict['label'])

        print("Size of model after quantization")
        quantized_model.print_size_of_model()

    if pars.weight_sharing:
        '''use_cuda = pars.use_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else 'cpu')

        weight_sharing_model = load_model(get_model(cuda=0), save_model_name)'''
        weight_sharing_model = apply_weight_sharing(model, bits=5) # TODO not smaller but faster?
        weight_sharing_model.print_size_of_model()
        weight_sharing_model.time_model_evaluation(valid_dict['index'], valid_dict['value'], valid_dict['label'])

    if pars.huffman_encoding:
        use_cuda = pars.use_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else 'cpu')

        huffmann_model = load_model(get_model(cuda=0), save_model_name)
        huffman_encode_model(model)
