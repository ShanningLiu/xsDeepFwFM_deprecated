import random, math, os
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

pd.options.mode.chained_assignment = None

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

bert_features = ['bert_' + str(i) for i in range(0, 768)]

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
        inputs[inputs.columns[i + len(dense_features) + 1]] = inputs[inputs.columns[i + len(dense_features) + 1]].map(col_map)

    if bert_data is not None:
        export_df = inputs.copy() # TODO copy is bad
        export_df[bert_features] = pd.DataFrame(bert_train)
        export_df[['label'] + dense_features + bert_features + sparse_features].to_csv(train_csv, sep=',', encoding='utf-8', index=False, header=False)
    else:
        inputs[['label'] + dense_features + sparse_features].to_csv(train_csv, sep=',', encoding='utf-8', index=False, header=False)

    # write feature_map file
    f_map = open(file_feature_map, 'w')
    for i in range(len(dense_features) + 1, len(list(inputs.columns))):
        for feature in feature_map[i]:
            if feature_map[i][feature] != 0:
                f_map.write(str(i) + ',' + feature + ',' + str(feature_map[i][feature]) + '\n')
    return feature_map


def generate_valid_csv(inputs, valid_csv, feature_map):
    for i, col_map in enumerate(feature_map[len(dense_features) + 1:]):
        inputs[inputs.columns[i + len(dense_features) + 1]] = inputs[inputs.columns[i + len(dense_features) + 1]].map(
            col_map)

    if bert_data is not None:
        export_df = inputs.copy()  # TODO copy is bad
        export_df[bert_features] = pd.DataFrame(bert_valid)
        export_df[['label'] + dense_features + bert_features + sparse_features].to_csv(valid_csv, sep=',',
                                                                                    encoding='utf-8', index=False,
                                                                                    header=False)
    else:
        inputs[['label'] + dense_features + sparse_features].to_csv(valid_csv, sep=',', encoding='utf-8', index=False,
                                                                    header=False)


def add_binary_labels(df, className):
    df['label'] = np.isfinite(df[className]).astype(int)
    return data


chunksize = 1 * 10 ** 3
bert_feature_file = "G:\\training_final_bert.csv"
category = 'like_engagement_timestamp'
data = None
for data in pd.read_csv("G:\\training_s.tsv", sep='\x01', encoding='utf8', chunksize=chunksize,
                   names=names, converters={'hashtags': lambda x: x.split('\t'),
                                            'present_media': lambda x: x.split('\t'),
                                            'present_links': lambda x: x.split('\t'),
                                            'present_domains': lambda x: x.split('\t')}):
    data = add_binary_labels(data, category)

    data['number_text_tokens'] = data['text_tokens'].apply(lambda x: str(x).count('\t') + 1)  # number of tokens
    data['number_hashtags'] = data['hashtags'].apply(lambda x: len(x) - 1)  # number of hashtags

    data['present_media'] = data['present_media'].apply(lambda x: len(x) - 1)
    data['present_links'] = data['present_links'].apply(lambda x: len(x) - 1)
    data['present_domains'] = data['present_domains'].apply(lambda x: len(x) - 1)

    # data[sparse_features] = data[sparse_features].fillna('-1', )
    # data[dense_features] = data[dense_features].fillna(0, )
    break

bert_data = None
for bert_data in pd.read_csv(bert_feature_file, sep=',', encoding='utf8', dtype=str, chunksize=chunksize, header=None):
        break

data = data[['label'] + dense_features + sparse_features]

mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

print('Split the original dataset into train and valid dataset.')
data_train, data_valid = train_test_split(data, test_size=0.2, random_state=42)
if bert_data is not None:
    bert_train, bert_valid = train_test_split(bert_data, test_size=0.2, random_state=42)  # is same split with same random state

# Not the best way, follow xdeepfm
print("Count freq in train")
freq_dict = cnt_freq_train(data)

print('Generate the feature map and impute the training dataset.')
feature_map = generate_feature_map_and_train_csv(data_train, 'train_twitter_pandas.csv', 'twitter_feature_map_pandas', freq_dict,
                                                 threshold=1)
print('Impute the valid dataset.')
generate_valid_csv(data_valid, 'valid_twitter_pandas.csv', feature_map) # TODO bert
