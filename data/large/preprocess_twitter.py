import random, math, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

random.seed(0)

pd.set_option('display.max_columns', 50)

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
        return str(int(math.log(float(x)) ** 2))  # log transformation to normalize numerical features
    else:
        return str(int(float(x)))


def cnt_freq_train(inputs):
    count_freq = []
    for i in range(len(list(inputs.columns))):
        count_freq.append({})
    for idx, line in inputs.iterrows():
        if idx % 1000000 == 0 and idx > 0:
            print(idx)
        for i in range(1, len(list(inputs.columns))):
            if i < len(dense_features) + 1:
                # line[i] = project_numeric(line[i])
                line[i] = scale(line[i])
            if line[i] not in count_freq[i]:
                count_freq[i][line[i]] = 0
            count_freq[i][line[i]] += 1
    return count_freq


def generate_feature_map_and_train_csv(inputs, train_csv, file_feature_map, freq_dict, threshold=8):
    feature_map = []
    for i in range(len(list(inputs.columns))):
        feature_map.append({})
    fout = open(train_csv, 'w')
    for idx, line in inputs.iterrows():
        if idx % 1000000 == 0 and idx > 0:
            print(idx)
        output_line = [str(line['label'])]
        for i in range(1, len(list(inputs.columns))):
            # map numerical features
            if i < len(dense_features) + 1:
                # line[i] = project_numeric(line[i])
                line[i] = scale(line[i])
                output_line.append(line[i])
            # handle categorical features
            elif freq_dict[i][line[i]] < threshold:
                output_line.append('0')
            elif line[i] in feature_map[i]:
                output_line.append(feature_map[i][line[i]])
            else:
                output_line.append(str(len(feature_map[i]) + 1))
                feature_map[i][str(line[i])] = str(len(feature_map[i]) + 1)
        output_line = ','.join(output_line)
        fout.write(output_line + '\n')

    # write feature_map file
    f_map = open(file_feature_map, 'w')
    for i in range(1, len(list(inputs.columns))):
        # only_one_zero_index = True
        for feature in feature_map[i]:
            # if feature_map[i][feature] == '0' and only_one_zero_index == False:
            #    continue
            f_map.write(str(i) + ',' + feature + ',' + feature_map[i][feature] + '\n')
            # if only_one_zero_index == True and feature_map[i][feature] == '0':
            #    only_one_zero_index = False
    return feature_map


def generate_valid_csv(inputs, valid_csv, feature_map):
    fout = open(valid_csv, 'w')
    for idx, line in inputs.iterrows():
        output_line = [str(line['label'])]
        for i in range(1, len(list(inputs.columns))):
            if i < len(dense_features) + 1:
                # line[i] = project_numeric(line[i])
                line[i] = scale(line[i])
                output_line.append(line[i])
            elif line[i] in feature_map[i]:
                output_line.append(feature_map[i][line[i]])
            else:
                output_line.append('0')
        output_line = ','.join(output_line)
        fout.write(output_line + '\n')


def add_binary_labels(data, className):
    data['label'] = 0
    data = data.assign(label=np.where(np.isnan(data[className]), data['label'], 1))

    return data


data = []
for chunk in pd.read_csv("C:\\Users\\AndreasPeintner\\Downloads\\training.tsv", sep='\x01', encoding='utf8',
                         names=names,
                         chunksize=1000 * 10 ** 3, converters={'hashtags': lambda x: x.split('\t'),
                                                             'present_media': lambda x: x.split('\t'),
                                                             'present_links': lambda x: x.split('\t'),
                                                             'present_domains': lambda x: x.split('\t')}):
    data = chunk
    break

category = 'like_engagement_timestamp'
data = add_binary_labels(data, category)

data['number_text_tokens'] = data['text_tokens'].apply(lambda x: str(x).count('\t') + 1)  # number of tokens
data['number_hashtags'] = data['hashtags'].apply(lambda x: len(x) - 1)  # number of hashtags

data['present_media'] = data['present_media'].apply(lambda x: len(x) - 1)
data['present_links'] = data['present_links'].apply(lambda x: len(x) - 1)
data['present_domains'] = data['present_domains'].apply(lambda x: len(x) - 1)

#data[sparse_features] = data[sparse_features].fillna('-1', )
#data[dense_features] = data[dense_features].fillna(0, )
data = data[['label'] + dense_features + sparse_features]

#mms = MinMaxScaler(feature_range=(0, 1))
#data[dense_features] = mms.fit_transform(data[dense_features])

print('Split the original dataset into train and valid dataset.')
data_train, data_valid = train_test_split(data, test_size=0.2)

#print(data_train)

# Not the best way, follow xdeepfm
print("count freq in train")
freq_dict = cnt_freq_train(data)

print('Generate the feature map and impute the training dataset.')
feature_map = generate_feature_map_and_train_csv(data_train, 'train_twitter.csv', 'twitter_feature_map', freq_dict,
                                                 threshold=8)
print('Impute the valid dataset.')
generate_valid_csv(data_valid, 'valid_twitter.csv', feature_map)
