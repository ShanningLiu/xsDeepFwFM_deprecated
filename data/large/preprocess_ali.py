import pandas as pd
from sklearn.utils import shuffle

ali_click = pd.read_csv('ali_click.csv', header=None, index_col=None, low_memory=False)
ali_click = shuffle(ali_click)


def cnt_freq_train(inputs):
    count_freq = []
    for col in inputs:
        count_freq.append(inputs[col].value_counts())

    return count_freq


def generate_feature_map_and_train_csv(inputs, freq_dict, file_feature_map):
    feature_map = []
    for freq in freq_dict:
        col_map = {}
        for idx, (key, value) in enumerate(freq.items()):
            col_map[key] = idx + 1

        feature_map.append(col_map)
    for i, col_map in enumerate(feature_map[2 + 1:]):
        inputs[inputs.columns[i + 2 + 1]] = inputs[inputs.columns[i + 2 + 1]].map(col_map)

    # write feature_map file
    f_map = open(file_feature_map, 'w')
    for i in range(3, 19):
        for feature in feature_map[i]:
            if feature_map[i][feature] != 0:
                f_map.write(str(i) + ',' + str(feature) + ',' + str(feature_map[i][feature]) + '\n')
    return feature_map


def generate_valid_csv(inputs, feature_map):
    for i, col_map in enumerate(feature_map[2 + 1:]):
        inputs[inputs.columns[i +2 + 1]] = inputs[inputs.columns[i +2 + 1]].map(col_map)


# no test data with labels online available
print('Split the orignal dataset into train and valid dataset.')
train_raw = ali_click.sample(frac=0.9, random_state=0, axis=0).reset_index(drop=True)
test = ali_click[~ali_click.index.isin(train_raw.index)]

valid = train_raw.sample(frac=0.5, random_state=0, axis=0).reset_index(drop=True)
train = train_raw[~train_raw.index.isin(valid.index)].reset_index(drop=True)

train.fillna(0, inplace=True)
valid.fillna(0, inplace=True)
test.fillna(0, inplace=True)
#

#
# valid = pd.read_csv('ali_valid.csv', header=None, index_col=None, low_memory=False)
# train = pd.read_csv('ali_train.csv', header=None, index_col=None, low_memory=False)
# test = pd.read_csv('ali_test.csv', header=None, index_col=None, low_memory=False)

#
# # Not the best way, follow xdeepfm
print('Count the frequency.')
freq_dict = cnt_freq_train(train)

#
print('Generate the feature map and impute the training dataset.')
feature_map = generate_feature_map_and_train_csv(train, freq_dict, 'ali_feature_map')

generate_valid_csv(valid, feature_map)
generate_valid_csv(test, feature_map)

train.fillna(0, inplace=True)
valid.fillna(0, inplace=True)
test.fillna(0, inplace=True)
#


train = shuffle(train)

valid = shuffle(valid)

test = shuffle(test)
#
valid.to_csv('ali_valid.csv', header=None, index=None)
train.to_csv('ali_train.csv', header=None, index=None)
test.to_csv('ali_test.csv', header=None, index=None)