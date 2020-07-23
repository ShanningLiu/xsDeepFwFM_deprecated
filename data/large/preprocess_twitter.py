import random, math, os
import dask
import dask.dataframe as dd
import dask.array as da
import pandas as pd
import numpy as np
from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import MinMaxScaler

random.seed(0)

names = ['text_tokens', 'hashtags', 'tweet_id', 'present_media', 'present_links', 'present_domains', 'tweet_type',
         'language', 'timestamp',
         'engaged_user_id', 'engagedfollower_count', 'engaged_following_count', 'engaged_verified',
         'engaged_account_creation_time',
         'engaging_user_id', 'engaging_follower_count', 'engaging_following_count', 'engaging_verified',
         'engaging_account_creation_time',
         'engagee_follows_engager', 'reply_engagement_timestamp', 'retweet_engagement_timestamp',
         'retweet_with_comment_engagement_timestamp', 'like_engagement_timestamp']

sparse_features = ['language', 'tweet_id', 'engaged_user_id', 'engaging_user_id',
                   'tweet_type']  # = categorical features

dense_features = ['number_text_tokens', 'number_hashtags', 'present_media', 'present_links', 'present_domains',
                  'timestamp', 'engagedfollower_count', 'engaged_following_count', 'engaged_verified',
                  'engaged_account_creation_time', 'engaging_follower_count', 'engaging_following_count',
                  'engaging_verified', 'engaging_account_creation_time',
                  'engagee_follows_engager']  # = numerical features


# https://github.com/WayneDW/AutoInt/blob/master/Dataprocess/Criteo/scale.py
def scale(x):
    if x == '':
        return '0'
    elif float(x) > 2:
        return int(math.log(float(x)) ** 2) # log transformation to normalize numerical features
    else:
        return int(float(x))


def cnt_freq_train(inputs):
    count_freq = []

    for dense_feature in dense_features:
        inputs[dense_feature] = inputs.apply(lambda x: scale(x[dense_feature]), axis=1, meta=(dense_feature, 'int64'))  # needed?

    for col in inputs.columns:
        count_freq.append(inputs[col].value_counts().compute())

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

    #for dense_feature in dense_features:
    #    inputs[dense_feature] = inputs.apply(lambda x: scale(x[dense_feature]), axis=1, meta=(dense_feature, 'int64'))
    mms = MinMaxScaler(feature_range=(0, 1))
    inputs[dense_features] = mms.fit_transform(inputs[dense_features])

    for i, col_map in enumerate(feature_map[len(dense_features) + 1:]):
        inputs[inputs.columns[i + len(dense_features) + 1]] = inputs[inputs.columns[i + len(dense_features) + 1]].map(
            col_map)
    inputs.compute().to_csv(train_csv, sep=',', encoding='utf-8', index=False, header=False)

    # write feature_map file
    f_map = open(file_feature_map, 'w')
    for i in range(len(dense_features) + 1, len(list(inputs.columns))):
        for feature in feature_map[i]:
            if feature_map[i][feature] != 0:
                f_map.write(str(i) + ',' + feature + ',' + str(feature_map[i][feature]) + '\n')
    return feature_map


def generate_valid_csv(inputs, valid_csv, feature_map):
    for dense_feature in dense_features:
        inputs[dense_feature] = inputs.apply(lambda x: scale(x[dense_feature]), axis=1, meta=(dense_feature, 'int64'))

    for i, col_map in enumerate(feature_map[len(dense_features) + 1:]):
        inputs[inputs.columns[i + len(dense_features) + 1]] = inputs[inputs.columns[i + len(dense_features) + 1]].map(
            col_map)

    inputs.compute().to_csv(valid_csv, sep=',', encoding='utf-8', index=False, header=False)


def add_binary_labels(df, className):
    df['label'] = da.isfinite(df[className]).astype(int)
    return data


data = dd.read_csv("G:\\training_s.tsv", sep='\x01', encoding='utf8',
                   names=names, converters={'hashtags': lambda x: x.split('\t'),
                                            'present_media': lambda x: x.split('\t'),
                                            'present_links': lambda x: x.split('\t'),
                                            'present_domains': lambda x: x.split('\t')})

category = 'like_engagement_timestamp'
data = add_binary_labels(data, category)

data['number_text_tokens'] = data['text_tokens'].apply(lambda x: str(x).count('\t') + 1, meta=('text_tokens', 'int64'))  # number of tokens
data['number_hashtags'] = data['hashtags'].apply(lambda x: len(x) - 1, meta=('hashtags', 'int64'))  # number of hashtags

data['present_media'] = data['present_media'].apply(lambda x: len(x) - 1, meta=('present_media', 'int64'))
data['present_links'] = data['present_links'].apply(lambda x: len(x) - 1, meta=('present_links', 'int64'))
data['present_domains'] = data['present_domains'].apply(lambda x: len(x) - 1, meta=('present_domains', 'int64'))

# data[sparse_features] = data[sparse_features].fillna('-1', )
# data[dense_features] = data[dense_features].fillna(0, )
data = data[['label'] + dense_features + sparse_features]

# mms = MinMaxScaler(feature_range=(0, 1))
# data[dense_features] = mms.fit_transform(data[dense_features])

print('Split the original dataset into train and valid dataset.')
data_train, data_valid = train_test_split(data, shuffle=True, test_size=0.2)

# print(data_train)

# Not the best way, follow xdeepfm
print("Count freq in train")
freq_dict = cnt_freq_train(data)

print('Generate the feature map and impute the training dataset.')
feature_map = generate_feature_map_and_train_csv(data_train, 'train_twitter_s.csv', 'twitter_feature_map_s', freq_dict,
                                                 threshold=8)
print('Impute the valid dataset.')
generate_valid_csv(data_valid, 'valid_twitter_s.csv', feature_map)
