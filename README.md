# DeepLight: Deep Lightweight Feature Interactions

Deploying the end-to-end deep factorization machines has a critical issue in prediction latency. To handle this issue, we study the acceleration of the prediction by conducting structural pruning for DeepFwFM, which ends up with 46X speed-ups without sacrifice of the state-of-the-art performance on Criteo dataset.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-sparse-deep-factorization-machine-for/click-through-rate-prediction-on-criteo)](https://paperswithcode.com/sota/click-through-rate-prediction-on-criteo?p=a-sparse-deep-factorization-machine-for)

Please refer to the [arXiv paper](https://arxiv.org/pdf/2002.06987.pdf) if you are interested. 

In this repository additional model compression and acceleration will be contucted. All on the Twitter dataset given by the RecSys 2020 Challenge.

## Environment

1. Python 3

2. PyTorch

3. Pandas

4. Sklearn

## Input Format

This implementation requires the input data in the following format:

- Xi: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
    - indi_j is the feature index of feature field j of sample i in the dataset
- Xv: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
    - vali_j is the feature value of feature field j of sample i in the dataset
    - vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
- y: target of each sample in the dataset (1/0 for classification, numeric number for regression)


## How to run the dense models

The folder already has a tiny dataset to test. You can run the following models through

LR: logistic regression
```bash
$ python main_all.py -use_fm 0 -use_fwfm 0 -use_deep 0 -use_lw 0 -use_logit 1 > ./logs/all_logistic_regression
```

FM: factorization machine

```bash
$ python main_all.py -use_fm 1 -use_fwfm 0 -use_deep 0 -use_lw 0 > ./logs/all_fm_vanilla
```

FwFM: field weighted factorization machine

```bash
$ python main_all.py -use_fm 0 -use_fwfm 1 -use_deep 0 -use_lw 0 > ./logs/all_fwfm_vanilla
```

DeepFM: deep factorization machine

```bash
$ python main_all.py -use_fm 1 -use_fwfm 0 -use_deep 1 -use_lw 0 > ./logs/all_deepfm_vanilla
```

NFM: factorization machine

```bash
$ python NFM.py > ./logs/all_nfm
```

xDeepFM: extreme factorization machine

You may try the link here https://github.com/Leavingseason/xDeepFM


## How to conduct structural pruning


The default code gives 0.8123 AUC if apply 90% sparsity on the DNN component and the field matrix R and apply 40% (90%x0.444) on the embeddings.

```bash
python main_all.py -l2 6e-7 -n_epochs 10 -warm 2 -prune 1 -sparse 0.90  -prune_deep 1 -prune_fm 1 -prune_r 1 -use_fwlw 1 -emb_r 0.444 -emb_corr 1. > ./logs/deepfwfm_l2_6e_7_prune_all_and_r_warm_2_sparse_0.90_emb_r_0.444_emb_corr_1
```__

## Useful python scripts

Pruning
```bash
python main_all.py -use_fm 0 -use_fwfm 1 -use_deep 1 -use_lw 1 -n_epochs 10 -dataset tiny-criteo -use_cuda 1 -prune 1 -l2 6e-7 -warm 2 -sparse 0.9 -prune_deep 1 -prune_fm 1 -prune_r 1 -use_fwlw 1 -emb_r 0.444 -emb_corr 1.
```

QR Embeddings
```bash
python main_all.py -use_fm 0 -use_fwfm 1 -use_deep 1 -use_lw 1 -use_fwlw 1 -use_cuda 1 -n_epochs 3 -dataset criteo -embedding_bag 1 -qr_flag 1
```

Quantization for sparse models
```bash
python quantization.py -use_deep 1 -use_fwfm 1 -n_epochs 3 -prune 1 -sparse 0.90 -use_fwlw 1 -save_model_name ./saved_models/full_pruned_DeepFwFM_l2_6e-07_sparse_0.9_seed_0 -dynamic_quantization 0 -quantization_aware 0 -static_quantization 1
```

Quantization for QR Embeddings
```bash
python quantization.py -use_deep 1 -use_fwfm 1 -use_lw 1 -use_fwlw 1 -n_epochs 3 -save_model_name ./saved_models/full_DeepFwFM_l2_3e-07_qr -dynamic_quantization 0 -quantization_aware 0 -static_quantization 1 -embedding_bag 1 -qr_flag 1
```

## Preprocess full Twitter dataset

To download the full dataset, you can use the link below
https://recsys-twitter.com/

Move the file to the *./data/large* folder

Move to the data folder and process the raw data.
```bash
$ python preprocess_twitter.py
```

When the dataset is ready, you need to change the files in main_all.py as follows
```py
twitter_num_feat_dim = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

train_dict = data_preprocess.read_data('./data/large/train_twitter.csv', './data/large/twitter_feature_map',
                                       twitter_num_feat_dim, feature_dim_start=1, dim=20)
test_dict = data_preprocess.read_data('./data/large/valid_twitter.csv', './data/large/twitter_feature_map',
                                      twitter_num_feat_dim, feature_dim_start=1, dim=20)

model = DeepFMs.DeepFMs(field_size=20,...)
```

Then run with following parameter:
```bash
$ python main_all.py -... -numerical=15
```


## Preprocess full Criteo dataset

The Criteo dataset has 2-class labels with 22 categorical features and 11 numerical features.

To download the full dataset, you can use the link below
http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/

Unzip the raw data and save it in ./data/large folder:
>> tar xvzf dac.tar.gz

Move to the data folder and process the raw data.
```bash
$ python preprocess_criteo.py
```

When the dataset is ready, you need to change the files in main_all.py as follows
```py
#result_dict = data_preprocess.read_data('./data/tiny_train_input.csv', './data/category_emb', criteo_num_feat_dim, feature_dim_start=0, dim=39)
#test_dict = data_preprocess.read_data('./data/tiny_test_input.csv', './data/category_emb', criteo_num_feat_dim, feature_dim_start=0, dim=39)
result_dict = data_preprocess.read_data('./data/large/train.csv', './data/large/criteo_feature_map', criteo_num_feat_dim, feature_dim_start=1, dim=39)
test_dict = data_preprocess.read_data('./data/large/valid.csv', './data/large/criteo_feature_map', criteo_num_feat_dim, feature_dim_start=1, dim=39)
```





## How to analyze the prediction latency

You need to download this repo: https://github.com/uestla/Sparse-Matrix before you start.

After the setup, you can change the directory in line-23 of the cpp file to your local dir.

```bash
cd latency
g++ criteo_latency.cpp  -o criteo.out
```


To avoid setting the environment, you can also consider to test the compiled file directly.

```bash
./criteo.out
```



## Acknowledgement

https://github.com/nzc/dnn_ctr
