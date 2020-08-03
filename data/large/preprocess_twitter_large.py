import math
import os
import random

random.seed(0)


def random_split(inputs, output1, valid):
    fout1 = open(output1, 'w')
    fout2 = open(valid, 'w')
    for line in open(inputs):
        if random.uniform(0, 1) < 0.9:
            fout1.write(line)
        else:
            fout2.write(line)
    fout1.close()
    fout2.close()


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
    for i in range(len(all_features)):
        count_freq.append({})

    with open(inputs, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            if idx % 1000000 == 0 and idx > 0:
                print(idx)
            line = line.strip()
            features = line.split("\x01")

            for feature, i in all_features_to_idx.items():

                if feature in dense_features:
                    if type(features[i]) is str:
                        features[i] = len(features[i].split('\t')) - 1
                    features[i] = scale(features[i])

                if features[i] not in count_freq[i]:
                    count_freq[i][features[i]] = 0

                count_freq[i][features[i]] += 1

    return count_freq


def generate_feature_map_and_train_csv(inputs, train_csv, file_feature_map, freq_dict, category, threshold=4):
    feature_map = []
    for i in range(len(all_features)):
        feature_map.append({})
    fout = open(train_csv, 'w')
    with open(inputs, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            line = line.strip()
            features = line.split("\x01")
            if idx % 1000000 == 0 and idx > 0:
                print(idx)
            # label
            output_line = ["1"] if features[labels_to_idx[category]] else ["0"]
            category_line = []
            for feature, i in all_features_to_idx.items():
                # map numerical features
                if feature in dense_features:
                    if type(features[i]) is str:
                        features[i] = len(features[i].split('\t')) - 1
                    features[i] = scale(features[i])
                    output_line.append(features[i])

                # handle categorical features
                elif freq_dict[i][features[i]] < threshold:
                    category_line.append('0')
                elif features[i] in feature_map[i]:
                    category_line.append(feature_map[i][features[i]])
                else:
                    category_line.append(str(len(feature_map[i]) + 1))
                    feature_map[i][features[i]] = str(len(feature_map[i]) + 1)
            output_line = output_line + category_line
            output_line = ','.join(output_line)
            fout.write(output_line + '\n')

    # write feature_map file
    f_map = open(file_feature_map, 'w')
    for i in range(len(all_features)):
        for feature in feature_map[i]:
            f_map.write(str(i) + ',' + feature + ',' + feature_map[i][feature] + '\n')
    return feature_map


def get_feature_size(fname):
    cnts = [0] * 40
    mins = [1] * 40
    maxs = [1] * 40
    dicts = []
    for i in range(40):
        dicts.append(set())
    for line in open(fname):
        line = line.strip().split('\t')
        for i in range(40):
            if line[i] not in dicts[i]:
                cnts[i] += 1
                dicts[i].add(line[i])
            try:
                mins[i] = min(mins[i], float(line[i]))
                maxs[i] = max(maxs[i], float(line[i]))
            except:
                print(line)
    print(cnts)
    print(mins)
    print(maxs)


def generate_valid_csv(inputs, valid_csv, feature_map, category):
    fout = open(valid_csv, 'w')
    with open(inputs, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            line = line.strip()
            features = line.split("\x01")
            # label
            output_line = ["1"] if features[labels_to_idx[category]] else ["0"]
            category_line = []
            for feature, i in all_features_to_idx.items():
                if feature in dense_features:
                    if type(features[i]) is str:
                        features[i] = len(features[i].split('\t')) - 1
                    features[i] = scale(features[i])
                    output_line.append(features[i])
                elif features[i] in feature_map[i]:
                    category_line.append(feature_map[i][features[i]])
                else:
                    category_line.append('0')
            output_line = output_line + category_line
            output_line = ','.join(output_line)
            fout.write(output_line + '\n')


all_features = ['text_tokens', 'hashtags', 'tweet_id', 'present_media', 'present_links', 'present_domains',
                'tweet_type',
                'language', 'tweet_timestamp',
                'engaged_user_id', 'engagedfollower_count', 'engaged_following_count', 'engaged_verified',
                'engaged_account_creation_time',
                'engaging_user_id', 'engaging_follower_count', 'engaging_following_count', 'engaging_verified',
                'engaging_account_creation_time',
                'engagee_follows_engager']

sparse_features = ['tweet_type', 'language', 'tweet_id', 'engaged_user_id', 'engaging_user_id']  # = categorical features

dense_features = ['text_tokens', 'hashtags', 'present_media', 'present_links', 'present_domains',
                  'tweet_timestamp', 'engagedfollower_count', 'engaged_following_count', 'engaged_verified',
                  'engaged_account_creation_time', 'engaging_follower_count', 'engaging_following_count',
                  'engaging_verified', 'engaging_account_creation_time',
                  'engagee_follows_engager']  # = numerical features

category = 'like_engagement_timestamp'

all_features_to_idx = dict(zip(all_features, range(len(all_features))))
labels_to_idx = {"reply_engagement_timestamp": 20, "retweet_engagement_timestamp": 21,
                 "retweet_with_comment_engagement_timestamp": 22,
                 "like_engagement_timestamp": 23}

file = 'C:\\Users\\AndreasPeintner\\Downloads\\training_s.tsv'
print('Split the original dataset into train and valid dataset.')
random_split(file, 'train1.tsv', 'valid.tsv')

# Not the best way, follow xdeepfm
print("Count freq in train")
freq_dict = cnt_freq_train(file)

print('Generate the feature map and impute the training dataset.')
feature_map = generate_feature_map_and_train_csv('train1.tsv', 'train_twitter_large.csv', 'twitter_large_feature_map',
                                                 freq_dict, category,
                                                 threshold=4)
print('Impute the valid dataset.')
generate_valid_csv('valid.tsv', 'valid_twitter_large.csv', feature_map, category)
print('Delete unnecessary files')
os.remove('valid.tsv')
os.remove('train1.tsv')

# get_feature_size('train_shuffle.csv')
