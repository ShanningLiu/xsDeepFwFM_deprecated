import random
import numpy as np
import torch
from torchsummary import summary
from tqdm import trange

from model import DeepFMs
from model.Datasets import Dataset, get_dataset
from utils.parameters import get_parser
from utils.util import get_model, load_model_dic, get_logger

from datetime import datetime
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
    if pars.emb_bag and not pars.qr_emb:
        save_model_name = save_model_name + '_emb_bag'
    if pars.qr_emb:
        save_model_name = save_model_name + '_qr'

    save_model_name = save_model_name + '_' + datetime.now().strftime("%Y%m%d%H%M%S")

    logger = get_logger(save_model_name[14:])
    logger.info(pars)

    logger.info("GET DATASET")
    field_size, train_dict, valid_dict, test_dict = get_dataset(pars)

    model = get_model(field_size=field_size, cuda=pars.use_cuda and torch.cuda.is_available(), feature_sizes=train_dict['feature_sizes'], pars=pars, logger=logger)
    #summary(model, [(train_dict['index'].shape[1], 1), (train_dict['value'].shape[1], )], dtypes=[torch.long, torch.float], device=torch.device("cpu"))

    if pars.use_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        #logger.info(torch.cuda.memory_summary(device=None, abbreviated=False))
        model = model.cuda()

    model.fit(train_dict['index'], train_dict['value'], train_dict['label'], valid_dict['index'],
              valid_dict['value'], valid_dict['label'],
              prune=pars.prune, prune_fm=pars.prune_fm, prune_r=pars.prune_r, prune_deep=pars.prune_deep,
              save_path=save_model_name, emb_r=pars.emb_r, emb_corr=pars.emb_corr, early_stopping=False)

    # measurements
    model = get_model(field_size=field_size, cuda=pars.time_on_cuda, feature_sizes=train_dict['feature_sizes'], pars=pars, logger=logger)
    model = load_model_dic(model, save_model_name, sparse=pars.prune)
    if pars.time_on_cuda:
        model = model.cuda()

    model.print_size_of_model()
    logger.info("TEST DATASET")
    #model.run_benchmark(test_dict['index'], test_dict['value'], test_dict['label'], batch_size=8192, cuda=pars.time_on_cuda)

    Xi = np.array(test_dict['index']).reshape((-1, model.field_size - model.num, 1))
    Xv = np.array(test_dict['value'])
    batch_xi = torch.autograd.Variable(torch.LongTensor(Xi[0:8192]))
    batch_xv = torch.autograd.Variable(torch.FloatTensor(Xv[0:8192]))
    mini_batch_xi = torch.autograd.Variable(torch.LongTensor(Xi[0:1]))
    mini_batch_xv = torch.autograd.Variable(torch.FloatTensor(Xv[0:1]))

    mini_batch_xi = mini_batch_xi.to(device='cpu', dtype=torch.long)
    mini_batch_xv = mini_batch_xv.to(device='cpu', dtype=torch.float)

    model.cpu()
    model.eval()

    with torch.no_grad(): # warmup
        _ = model(batch_xi, batch_xv)

    with torch.no_grad():
        with torch.autograd.profiler.profile(use_cuda=False) as prof:
            _ = model(mini_batch_xi, mini_batch_xv)
    logger.info(prof.key_averages().table(sort_by="self_cpu_time_total"))

