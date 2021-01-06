import torch
from model import DeepFMs
import numpy as np, gc
import logging
import sys
import tqdm

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

def get_logger(filename=None):
    root = logging.getLogger('xsDeepFwFM')
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    if filename:
        file_handler = logging.FileHandler(filename='./logs/' + filename + '.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    #root.addHandler(TqdmLoggingHandler(level=logging.DEBUG))

    root.propagate = False

    return root


def load_model_dic(model, model_file, sparse=False):
    state_dict = torch.load(model_file)
    if sparse:
        model.load_state_dict(state_dict, strict=False)
        for name, param in model.named_parameters():
            if 'linear' in name and 'weight' in name:
                param.to_sparse()
    else:
        model.load_state_dict(state_dict)
    # model.to('cpu')
    return model


def get_model(cuda, feature_sizes, pars, dynamic_quantization=False, static_quantization=False,
              quantization_aware=False, field_size=39, deep_nodes=400, h_depth=3, logger=None):
    return DeepFMs.DeepFMs(field_size=field_size, feature_sizes=feature_sizes,
                           embedding_size=pars.embedding_size, n_epochs=pars.n_epochs,
                           verbose=False, use_cuda=cuda, use_fm=pars.use_fm, use_fwfm=pars.use_fwfm,
                           use_ffm=pars.use_ffm, use_deep=pars.use_deep,
                           batch_size=pars.batch_size, learning_rate=pars.learning_rate, weight_decay=pars.l2,
                           momentum=pars.momentum, sparse=pars.sparse, warm=pars.warm,
                           h_depth=h_depth, deep_nodes=deep_nodes, num_deeps=pars.num_deeps,
                           numerical=pars.numerical, use_lw=pars.use_lw, use_fwlw=pars.use_fwlw,
                           use_logit=pars.use_logit, random_seed=pars.random_seed,
                           quantization_aware=quantization_aware, dynamic_quantization=dynamic_quantization,
                           static_quantization=static_quantization, loss_type=pars.loss_type,
                           embedding_bag=pars.emb_bag,
                           qr_flag=pars.qr_emb, qr_operation=pars.qr_operation, qr_collisions=pars.qr_collisions,
                           qr_threshold=pars.qr_threshold, logger=logger)


def save_memory(df):
    features = df.columns
    for i in range(df.shape[1]):
        if df.dtypes[i] == 'uint8':
            df[features[i]] = df[features[i]].astype(np.int8)
            gc.collect()
        elif df.dtypes[i] == 'bool':
            df[features[i]] = df[features[i]].astype(np.int8)
            gc.collect()
        elif df.dtypes[i] == 'uint32':
            df[features[i]] = df[features[i]].astype(np.int32)
            gc.collect()
        elif df.dtypes[i] == 'int64':
            df[features[i]] = df[features[i]].astype(np.int32)
            gc.collect()
        elif df.dtypes[i] == 'float64':
            df[features[i]] = df[features[i]].astype(np.float32)
            gc.collect()

    gc.collect()

    return df
