import random
import argparse
import sys

import numpy as np
from sklearn.metrics import accuracy_score

from model import DeepFMs
from utils import data_preprocess
from utils.parameters import get_parser
from utils.util import get_model, load_model_dic, get_logger
from model.Datasets import Dataset, get_dataset

import torch
from torchsummary import summary

if __name__ == '__main__':
    parser = get_parser()
    pars = parser.parse_args()

    logger = get_logger('Quantization')
    logger.info(pars)

    field_size, train_dict, valid_dict, test_dict = get_dataset(pars)

    if not pars.save_model_path:
        logger.info("no model path given: -save_model_path")
        sys.exit()

    model = get_model(field_size=field_size, cuda=pars.use_cuda and torch.cuda.is_available(),
                      feature_sizes=train_dict['feature_sizes'], pars=pars, logger=logger)
    model = load_model_dic(model, pars.save_model_path, sparse=pars.prune)

    #summary(model, [(train_dict['index'].shape[1], 1), (train_dict['value'].shape[1], )], dtypes=[torch.long, torch.float], device=torch.device("cpu"))

    if pars.use_cuda:
        model.cuda()

    logger.info('Original model:')
    model.print_size_of_model()
    model.run_benchmark(test_dict['index'], test_dict['value'], test_dict['label'], cuda=pars.use_cuda)

    # dynamic quantization (no CUDA allowed and dynamic after training)
    # https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html
    # not really best for our use case:
    # This is used for situations where the model execution time is dominated by loading weights from memory rather than computing the matrix multiplications.
    # This is true for for LSTM and Transformer type models with small batch size.
    if pars.dynamic_quantization:
        quantized_model = load_model_dic(
            get_model(field_size=field_size, cuda=0, feature_sizes=train_dict['feature_sizes'],
                      dynamic_quantization=True, pars=pars, logger=logger), pars.save_model_path,
            sparse=pars.prune)  # no logger allowed here

        quantized_model.eval()
        quantized_model = torch.quantization.quantize_dynamic(quantized_model, {torch.nn.Linear}, dtype=torch.qint8)

        logger.info("Dynamic Quantization model:")
        q = quantized_model.print_size_of_model()
        # logger.info("\t{0:.2f} times smaller".format(f / q))
        # logger.info(quantized_model)

        quantized_model.run_benchmark(test_dict['index'], test_dict['value'], test_dict['label'])

        torch.save(quantized_model.state_dict(), pars.save_model_path + '_dynamic_quant')

    # most commonly used form of quantization
    # embedding quantization in pytorch 1.7.1
    # Support for FP16 quantization
    # Embedding and EmbeddingBag quantization (8-bit + partial support for 4-bit)
    # https://discuss.pytorch.org/t/is-it-planned-to-support-nn-embeddings-quantization/89154
    # https://github.com/pytorch/pytorch/issues/41396
    if pars.static_quantization:  # https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
        quantized_model = load_model_dic(
            get_model(field_size=field_size, cuda=0, feature_sizes=train_dict['feature_sizes'],
                      static_quantization=True, pars=pars, logger=logger), pars.save_model_path, sparse=pars.prune)
        quantized_model.eval()

        quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        #quantized_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        #quantized_model.qconfig = torch.quantization.default_qconfig

        quantized_model = torch.quantization.fuse_modules(quantized_model,
                                                     [['net_1_linear_1', 'net_1_linear_1_relu'],
                                                      ['net_1_linear_2', 'net_1_linear_2_relu'],
                                                      ['net_1_linear_3', 'net_1_linear_3_relu']])

        torch.quantization.prepare(quantized_model, inplace=True)

        #logger.info(quantized_model)

        # Calibrate
        quantized_model.static_calibrate = True
        calibration_size = quantized_model.batch_size * 5
        Xi = train_dict['index'][:calibration_size]
        Xv = train_dict['value'][:calibration_size]
        y = train_dict['label'][:calibration_size]
        Xi = np.array(Xi).reshape((-1, quantized_model.field_size - quantized_model.num, 1))
        Xv = np.array(Xv)
        y = np.array(y)
        x_size = Xi.shape[0]
        quantized_model.eval()
        quantized_model.eval_by_batch(Xi, Xv, y, x_size)
        logger.info('Post Static Quantization: Calibration done')

        # Convert to quantized model
        quantized_model.static_calibrate = False
        torch.quantization.convert(quantized_model, inplace=True)

        # logger.info(quantized_model)
        logger.info("Post Static Quantization model:")
        quantized_model.print_size_of_model()
        quantized_model.run_benchmark(test_dict['index'], test_dict['value'], test_dict['label'], cuda=pars.use_cuda)

        torch.save(quantized_model.state_dict(), pars.save_model_path + '_static_quant')

    # QAT supports CUDA with fake quantization: https://pytorch.org/docs/stable/quantization.html
    # convertion happens in evaluation method
    if pars.quantization_aware:
        quantized_model = get_model(field_size=field_size, cuda=1, feature_sizes=train_dict['feature_sizes'],
                                    quantization_aware=True, pars=pars, logger=logger)
        quantized_model.cuda()
        quantized_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

        logger.info(quantized_model.qconfig)

        torch.quantization.prepare(quantized_model, inplace=True)
        #logger.info(quantized_model)
        quantized_model.fit(train_dict['index'], train_dict['value'], train_dict['label'], valid_dict['index'],
                            valid_dict['value'], valid_dict['label'],
                            prune=pars.prune, prune_fm=pars.prune_fm, prune_r=pars.prune_r, prune_deep=pars.prune_deep,
                            emb_r=pars.emb_r, emb_corr=pars.emb_corr,
                            quantization_aware=True)

        torch.save(quantized_model.state_dict(), pars.save_model_path + '_quant_aware')

        logger.info("Quantization Aware model:")
        state_dict = torch.load(pars.save_model_path + '_quant_aware')
        quantized_model.load_state_dict(state_dict)
        quantized_model.to('cpu')
        quantized_model.eval()

        quantized_model.quantization_aware = True
        quantized_model.use_cuda = False

        q = quantized_model.print_size_of_model()
        # logger.info("\t{0:.2f} times smaller".format(f / q))
        quantized_model.run_benchmark(test_dict['index'], test_dict['value'], test_dict['label'], cuda=False)
