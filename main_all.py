import random
import numpy as np
import torch
from torchsummary import summary
from tqdm import trange

from model import DeepFMs
from model.Datasets import Dataset, get_dataset
from utils.parameters import get_parser
from utils.util import get_model, load_model_dic, get_logger

import os

parser = get_parser()
pars = parser.parse_args()

if __name__ == '__main__':
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    np.random.seed(pars.random_seed)
    random.seed(pars.random_seed)
    torch.manual_seed(pars.random_seed)
    torch.cuda.manual_seed(pars.random_seed)

    save_model_name = './saved_models/' + pars.c + '_l2_' + str(pars.l2) + '_dt_' + pars.dataset

    if pars.prune:
        save_model_name = save_model_name + '_sparse_' + str(pars.sparse) + '_seed_' + str(pars.random_seed)
    if pars.qr_flag:
        save_model_name = save_model_name + '_qr'
    if pars.md_flag:
        save_model_name = save_model_name + '_md'

    logger = get_logger(save_model_name[14:])
    logger.info(pars)

    field_size, train_dict, valid_dict = get_dataset(pars)

    model = get_model(field_size=field_size, cuda=pars.use_cuda and torch.cuda.is_available(), feature_sizes=train_dict['feature_sizes'], pars=pars, logger=logger)
    if pars.use_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        model = model.cuda()

    model.fit(train_dict['index'], train_dict['value'], train_dict['label'], valid_dict['index'],
              valid_dict['value'], valid_dict['label'],
              prune=pars.prune, prune_fm=pars.prune_fm, prune_r=pars.prune_r, prune_deep=pars.prune_deep,
              save_path=save_model_name, emb_r=pars.emb_r, emb_corr=pars.emb_corr, early_stopping=False)

    # measurement
    time_on_cuda = False

    model = get_model(field_size=field_size, cuda=time_on_cuda, feature_sizes=train_dict['feature_sizes'], pars=pars, logger=logger)
    model = load_model_dic(model, save_model_name)
    if time_on_cuda:
        model = model.cuda()

    model.print_size_of_model()
    model.run_benchmark(valid_dict['index'], valid_dict['value'], valid_dict['label'], cuda=time_on_cuda)
