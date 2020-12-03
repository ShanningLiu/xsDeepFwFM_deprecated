import random, math, os
import pandas as pd, numpy as np, gc
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils.util import save_memory

pd.options.mode.chained_assignment = None

random.seed(0)

feature_names = ['timestamp', 'a_follower_count', 'a_following_count', 'a_is_verified',
                 'a_account_creation', 'b_follower_count', 'b_following_count',
                 'b_is_verified', 'b_account_creation', 'b_follows_a', 'reply',
                 'retweet', 'retweet_comment', 'like', 'id', 'language', 'tweet_type',
                 'media', 'tweet_id', 'a_user_id', 'b_user_id', 'len_hashtags',
                 'len_domains', 'len_links', 'domains', 'links', 'hashtags', 'tr',
                 'dt_day', 'dt_dow', 'dt_hour', 'a_count_combined',
                 'a_user_fer_count_delta_time', 'a_user_fing_count_delta_time',
                 'a_user_fering_count_delta_time', 'a_user_fing_count_mode',
                 'a_user_fer_count_mode', 'a_user_fering_count_mode', 'count_ats',
                 'count_char', 'count_words', 'tw_hash', 'tw_freq_hash', 'tw_first_word',
                 'tw_second_word', 'tw_last_word', 'tw_llast_word', 'tw_len', 'tw_hash0',
                 'tw_hash1', 'tw_rt_uhash']

sparse_features = ['a_is_verified', 'b_is_verified', 'b_follows_a', 'id',
                   'language', 'tweet_type', 'media', 'tweet_id', 'a_user_id', 'b_user_id', 'domains', 'links',
                   'hashtags', 'tr', 'dt_day', 'dt_dow', 'dt_hour', 'a_count_combined', 'a_user_fer_count_delta_time',
                   'a_user_fing_count_delta_time',
                   'a_user_fering_count_delta_time', 'a_user_fing_count_mode',
                   'a_user_fer_count_mode', 'a_user_fering_count_mode', 'count_ats',
                   'count_char', 'count_words', 'tw_hash', 'tw_freq_hash', 'tw_first_word',
                   'tw_second_word', 'tw_last_word', 'tw_llast_word', 'tw_hash0',
                   'tw_hash1', 'tw_rt_uhash']  # = categorical features

dense_features = ['timestamp', 'a_follower_count', 'a_following_count', 'a_account_creation',
                  'b_follower_count', 'b_following_count', 'b_account_creation',
                  'len_hashtags', 'len_domains', 'len_links', 'tw_len']  # = numerical features

label_names = ['reply', 'retweet', 'retweet_comment', 'like']

def cnt_freq_train(inputs):
    count_freq = []

    for col in label_names + dense_features + sparse_features:
        count_freq.append(inputs[col].value_counts())

    return count_freq


def generate_feature_map_and_train_csv(inputs, train_csv, file_feature_map, freq_dict, threshold=8):
    feature_map = []
    for freq in freq_dict:
        col_map = {}
        for idx, (key, value) in enumerate(freq.items()):
            if value >= threshold:
                col_map[key] = idx + 1
            else:
                col_map[key] = 0

        feature_map.append(col_map)

    for i, col_map in enumerate(feature_map[len(dense_features) + 1:]):
        inputs[inputs.columns[i + len(dense_features) + 1]] = inputs[inputs.columns[i + len(dense_features) + 1]].map(
            col_map)

    inputs.fillna(0, inplace=True)
    inputs = save_memory(inputs)
    inputs[label_names + dense_features + sparse_features].to_parquet(train_csv)

    # write feature_map file
    f_map = open(file_feature_map, 'w')
    for i in range(len(dense_features) + 1, len(list(inputs.columns))):
        for feature in feature_map[i]:
            if feature_map[i][feature] != 0:
                f_map.write(str(i) + ',' + str(feature) + ',' + str(feature_map[i][feature]) + '\n')
    return feature_map


def generate_test_csv(inputs, valid_csv, feature_map):
    for i, col_map in enumerate(feature_map[len(dense_features) + 1:]):
        inputs[inputs.columns[i + len(dense_features) + 1]] = inputs[inputs.columns[i + len(dense_features) + 1]].map(
            col_map)

    inputs.fillna(0, inplace=True)
    inputs = save_memory(inputs)
    inputs[label_names + dense_features + sparse_features].to_parquet(valid_csv)



train = pd.read_parquet('./train_final_s.parquet')
valid = pd.read_parquet('./valid_final_s.parquet')
test = pd.read_parquet('./test_final_s.parquet')

data = pd.concat((train, valid, test), sort=False)

data.fillna(0, inplace=True)

data = data[label_names + dense_features + sparse_features]

mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

print('Split dataset')
train = data.loc[(data.tr == 0)]
valid = data.loc[(data.tr == 1)]
test = data.loc[(data.tr == 2)]

print(train.shape, valid.shape, test.shape)
print(train.head())

del data

# Not the best way, follow xdeepfm
print("Count freq in train")
freq_dict = cnt_freq_train(train)

print('Generate the feature map and impute the training dataset.')
feature_map = generate_feature_map_and_train_csv(train, 'twitter_train_s.parquet', 'twitter_feature_map_s', freq_dict,
                                                 threshold=8)
print('Impute the valid dataset.')
generate_test_csv(valid, 'twitter_valid_s.parquet', feature_map)

print('Impute the test dataset.')
generate_test_csv(test, 'twitter_test_s.parquet', feature_map)
