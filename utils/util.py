import torch
from model import DeepFMs

def load_model_dic(model, model_file, sparse=False):
    state_dict = torch.load(model_file)
    if sparse:
        model.load_state_dict(state_dict, strict=False)
        for name, param in model.named_parameters():
            if 'linear' in name and 'weight' in name:
                param.to_sparse()
    else:
        model.load_state_dict(state_dict)
    #model.to('cpu')
    return model

def get_model(cuda, feature_sizes, pars, dynamic_quantization=False, static_quantization=False, quantization_aware=False, field_size=39, deep_nodes=400, h_depth=3):
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
                           static_quantization=static_quantization, loss_type=pars.loss_type, embedding_bag=pars.embedding_bag,
                           qr_flag=pars.qr_flag, qr_operation=pars.qr_operation, qr_collisions=pars.qr_collisions,
                           qr_threshold=pars.qr_threshold, md_flag=pars.md_flag, md_threshold=pars.md_threshold)
