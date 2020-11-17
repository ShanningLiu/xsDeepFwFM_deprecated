import random, math, os
import pandas as pd, numpy as np, gc
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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

    df[sparse_features] = df[sparse_features].astype(int)

    return df

def cnt_freq_train(inputs):
    count_freq = []

    for col in ['label'] + dense_features + sparse_features:
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
    inputs[['label'] + dense_features + sparse_features].to_csv(train_csv, sep=',', encoding='utf-8', index=False,
                                                                header=False)

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
    inputs[['label'] + dense_features + sparse_features].to_csv(valid_csv, sep=',', encoding='utf-8', index=False,
                                                                header=False)


category = 'like'
data = pd.read_parquet('data-final-small.parquet')
data = save_memory(data)
data = data[:100000]
print(data.columns)

for label in label_names:
    if label != category:
        data = data.drop(columns=[label])

data = data[[category] + dense_features + sparse_features]
data = data.rename(columns={category: 'label'})
print(data.columns)

print(data.dtypes)

mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

print('Split dataset')
# train = data.loc[(data.tr == 0)]
# valid = data.loc[(data.tr == 1)]
# test = data.loc[(data.tr == 2)]
train = data[:70000]
valid = data[70000:90000]
test = data[90000:]
print(train.shape, valid.shape, test.shape)

print(train.head())

# Not the best way, follow xdeepfm
print("Count freq in train")
freq_dict = cnt_freq_train(data)

print('Generate the feature map and impute the training dataset.')
feature_map = generate_feature_map_and_train_csv(train, 'twitter_train.csv', 'twitter_feature_map', freq_dict,
                                                 threshold=4)
print('Impute the valid dataset.')
generate_test_csv(valid, 'twitter_valid.csv', feature_map)

print('Impute the test dataset.')
generate_test_csv(test, 'twitter_test.csv', feature_map)
