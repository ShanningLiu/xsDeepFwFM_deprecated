import torch
from model import DeepFMs

def load_model(model, model_file):
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    #model.to('cpu')
    return model


def get_model(cuda, feature_sizes, pars, dynamic_quantization=False, static_quantization=False, quantization_aware=False, field_size=39):
    return DeepFMs.DeepFMs(field_size=field_size, feature_sizes=feature_sizes,
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
