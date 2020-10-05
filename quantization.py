import random
import argparse
import sys

import numpy as np
from sklearn.metrics import accuracy_score

from model import DeepFMs
from utils import data_preprocess
from utils.parameters import getParser
from utils.util import get_model, load_model_dic

import torch

parser = getParser()
pars = parser.parse_args()

criteo_num_feat_dim = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
field_size = 39
'''train_dict = data_preprocess.read_data('./data/large/train_criteo_s.csv', './data/large/criteo_feature_map_s',
                                       criteo_num_feat_dim, feature_dim_start=1, dim=39)'''
train_dict = data_preprocess.get_feature_sizes('./data/large/full_criteo_feature_map',
                                       criteo_num_feat_dim, feature_dim_start=1, dim=39)
valid_dict = data_preprocess.read_data('./data/large/full_valid_criteo.csv', './data/large/full_criteo_feature_map',
                                       criteo_num_feat_dim, feature_dim_start=1, dim=39)

if not pars.save_model_name:
    print("no model path given: -save_model_name")
    sys.exit()



model = get_model(cuda=0, feature_sizes=train_dict['feature_sizes'], pars=pars)
model = load_model_dic(model, pars.save_model_name, sparse=True)
print('Original model:')
f = model.print_size_of_model()
test_batch = model.batch_size * 4
model.time_model_evaluation(valid_dict['index'][:test_batch], valid_dict['value'][:test_batch], valid_dict['label'][:test_batch])

# quantization (no CUDA allowed and dynamic after training)
# https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html
# not really best for our use case:
# This is used for situations where the model execution time is dominated by loading weights from memory rather than computing the matrix multiplications.
# This is true for for LSTM and Transformer type models with small batch size.
if pars.dynamic_quantization:
    quantized_model = load_model_dic(get_model(cuda=0, feature_sizes=train_dict['feature_sizes'], dynamic_quantization=True, pars=pars), pars.save_model_name)

    quantized_model.eval()
    quantized_model = torch.quantization.quantize_dynamic(quantized_model, {torch.nn.Linear}, dtype=torch.qint8)

    print("Dynamic Quantization model:")
    q = quantized_model.print_size_of_model()
    #print("\t{0:.2f} times smaller".format(f / q))
    #print(quantized_model)

    quantized_model.time_model_evaluation(valid_dict['index'][:test_batch], valid_dict['value'][:test_batch], valid_dict['label'][:test_batch])

    torch.save(quantized_model.state_dict(), pars.save_model_name + '_dynamic_quant')

# most commonly used form of quantization
if pars.static_quantization:  # https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
    quantized_model = load_model_dic(get_model(cuda=0, feature_sizes=train_dict['feature_sizes'], static_quantization=True, use_deep=pars.use_deep, pars=pars), pars.save_model_name)
    quantized_model.eval()

    quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(quantized_model, inplace=True)

    #print(quantized_model)

    # Calibrate
    quantized_model.static_calibrate = True
    calibration_size = quantized_model.batch_size * 10
    Xi = valid_dict['index'][:calibration_size]
    Xv = valid_dict['value'][:calibration_size]
    y = valid_dict['label'][:calibration_size]
    Xi = np.array(Xi).reshape((-1, quantized_model.field_size - quantized_model.num, 1))
    Xv = np.array(Xv)
    y = np.array(y)
    x_size = Xi.shape[0]
    quantized_model.eval()
    quantized_model.eval_by_batch(Xi, Xv, y, x_size)
    print('Post Static Quantization: Calibration done')

    # Convert to quantized model
    quantized_model.static_calibrate = False
    torch.quantization.convert(quantized_model, inplace=True)
    #print(quantized_model)
    print("Post Static Quantization model:")
    q = quantized_model.print_size_of_model()
    #print("\t{0:.2f} times smaller".format(f / q))

    quantized_model.time_model_evaluation(valid_dict['index'][:test_batch], valid_dict['value'][:test_batch], valid_dict['label'][:test_batch])

    torch.save(quantized_model.state_dict(), pars.save_model_name + '_static_quant')

# QAT supports CUDA with fake quantization: https://pytorch.org/docs/stable/quantization.html
# convertion happens in evaluation method
if pars.quantization_aware: 
    quantized_model = get_model(cuda=1, feature_sizes=train_dict['feature_sizes'], quantization_aware=True, pars=pars)
    quantized_model.cuda()
    quantized_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    print(quantized_model.qconfig)

    torch.quantization.prepare(quantized_model, inplace=True)
    print(quantized_model)
    quantized_model.fit(train_dict['index'], train_dict['value'], train_dict['label'], valid_dict['index'],
              valid_dict['value'], valid_dict['label'],
              prune=pars.prune, prune_fm=pars.prune_fm, prune_r=pars.prune_r, prune_deep=pars.prune_deep, emb_r=pars.emb_r, emb_corr=pars.emb_corr,
              quantization_aware=True)

    torch.save(quantized_model.state_dict(), pars.save_model_name + '_quant_aware')

    print("Quantization Aware model:")
    state_dict = torch.load(pars.save_model_name + '_quant_aware')
    quantized_model.load_state_dict(state_dict)
    quantized_model.to('cpu')
    quantized_model.eval()

    quantized_model.quantization_aware = True
    quantized_model.use_cuda = False

    q = quantized_model.print_size_of_model()
    #print("\t{0:.2f} times smaller".format(f / q))
    quantized_model.time_model_evaluation(valid_dict['index'][:test_batch], valid_dict['value'][:test_batch], valid_dict['label'][:test_batch], cuda=False)