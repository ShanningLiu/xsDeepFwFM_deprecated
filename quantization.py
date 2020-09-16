import random
import argparse
import sys

import numpy as np

from model import DeepFMs
from utils import data_preprocess
from utils.parameters import getParser
from utils.util import get_model, load_model

import torch

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

model = get_model(cuda=0, feature_sizes=train_dict['feature_sizes'], pars=pars)
model = load_model(model, pars.save_model_name)
print('Original model:')
f = model.print_size_of_model()
model.time_model_evaluation(valid_dict['index'], valid_dict['value'], valid_dict['label'])

# quantization (no CUDA allowed and dynamic after training)
# https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html
# not really best for our use case:
# This is used for situations where the model execution time is dominated by loading weights from memory rather than computing the matrix multiplications.
# This is true for for LSTM and Transformer type models with small batch size.
if pars.dynamic_quantization:
    quantized_model = load_model(get_model(cuda=0, feature_sizes=train_dict['feature_sizes'], dynamic_quantization=True, pars=pars), pars.save_model_name)

    quantized_model.eval()
    print(quantized_model.field_cov.weight)
    quantized_model = torch.quantization.quantize_dynamic(quantized_model, {torch.nn.Linear}, dtype=torch.qint8) # TODO float16 for better loss and acc; if available torch.nn.Embedding, torch.nn.Dropout and

    #print(torch.dequantize(quantized_model.field_cov.weight()))
    #print(quantized_model)
    print("Dynamic Quantization model:")
    q = quantized_model.print_size_of_model()
    print("\t{0:.2f} times smaller".format(f / q))

    quantized_model.time_model_evaluation(valid_dict['index'], valid_dict['value'], valid_dict['label'])

    torch.save(quantized_model.state_dict(), pars.save_model_name + '_dynamic_quant')

# most commonly used form of quantization
if pars.static_quantization:  # https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
    quantized_model = load_model(get_model(cuda=0, feature_sizes=train_dict['feature_sizes'], static_quantization=True, pars=pars), pars.save_model_name)
    quantized_model.eval()

    quantized_model.qconfig = torch.quantization.default_qconfig
    print(quantized_model.qconfig)
    torch.quantization.prepare(quantized_model, inplace=True)

    # Calibrate
    #quantized_model.time_model_evaluation(train_dict['index'][:10000], train_dict['value'][:10000], train_dict['label'][:10000])
    #print('Post Static Quantization: Calibration done')

    # Convert to quantized model
    torch.quantization.convert(quantized_model, inplace=True)
    print("Post Static Quantization model:")
    q = quantized_model.print_size_of_model()
    print("\t{0:.2f} times smaller".format(f / q))

    quantized_model.time_model_evaluation(valid_dict['index'], valid_dict['value'], valid_dict['label'])

    torch.save(quantized_model.state_dict(), pars.save_model_name + '_static_quant')

# not ready now to use: https://pytorch.org/docs/stable/quantization.html
'''
if pars.quantization_aware: 
    quantized_model = get_model(cuda=0, feature_sizes=train_dict['feature_sizes'], quantization_aware=True, pars=pars)
    quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    print(quantized_model.qconfig)

    torch.quantization.prepare(quantized_model, inplace=True)

    #quantized_model = quantized_model.cuda()
    quantized_model.fit(train_dict['index'], train_dict['value'], train_dict['label'], valid_dict['index'],
              valid_dict['value'], valid_dict['label'],
              prune=pars.prune, prune_fm=pars.prune_fm, prune_r=pars.prune_r, prune_deep=pars.prune_deep,
              save_path=pars.save_model_name + '_quant_aware', emb_r=pars.emb_r, emb_corr=pars.emb_corr,
              quantization_aware=True)

    print("Quantization Aware model:")
    quantized_model = get_model(cuda=False, feature_sizes=train_dict['feature_sizes'], quantization_aware=True, pars=pars)
    quantized_model = load_model(quantized_model, pars.save_model_name + '_quant_aware')
    quantized_model = quantized_model.cuda()
    quantized_model.print_size_of_model()
    quantized_model.time_model_evaluation(valid_dict['index'], valid_dict['value'], valid_dict['label'])
    '''