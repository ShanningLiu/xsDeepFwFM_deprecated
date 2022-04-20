import pandas as pd
from sklearn.utils import shuffle


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
    for i, col_map in enumerate(feature_map[1 + 1:]):
        inputs[inputs.columns[i + 1 + 1]] = inputs[inputs.columns[i + 1 + 1]].map(col_map)

    # write feature_map file
    f_map = open(file_feature_map, 'w')
    for i in range(2, 24):
        for feature in feature_map[i]:
            if feature_map[i][feature] != 0:
                f_map.write(str(i) + ',' + str(feature) + ',' + str(feature_map[i][feature]) + '\n')
    return feature_map


def generate_valid_csv(inputs, feature_map):
    for i, col_map in enumerate(feature_map[1 + 1:]):
        inputs[inputs.columns[i +1 + 1]] = inputs[inputs.columns[i +1 + 1]].map(col_map)


# # read raw dataset
# avazu = pd.read_csv('train', sep=',', header=None, low_memory=False)
# avazu.columns = ['y%s' % i for i in range(1, avazu.shape[1] + 1)]
# print(avazu.head())
#
# first_column = avazu.pop('y2')
# avazu.insert(0, 'y2', first_column)
# print(avazu.head())
#
# hours = avazu.pop('y3')
# avazu.insert(1, 'y3', hours)
# print(avazu.head())

#

# test = avazu.sample(frac=0.1, random_state=0, axis=0).reset_index(drop=True)
# avazu.fillna(0, inplace=True)
# avazu.to_csv('avazu_after_idx.csv', header=None, index=None)

avazu = pd.read_csv('avazu_after_idx.csv', header=None, index_col=None, low_memory=True, skiprows = 1)
print(avazu.head())
# avazu.drop([0,1,2])
print(avazu.head())

# train = pd.read_csv('avazu_train.csv', header=None, index_col=None, low_memory=False)
#
# # Not the best way, follow xdeepfm
print('Count the frequency.')
avazu.fillna(0, inplace=True)
avazu = shuffle(avazu)
freq_dict = cnt_freq_train(avazu)

#
print('Generate the feature map and impute the training dataset.')
feature_map = generate_feature_map_and_train_csv(avazu, freq_dict, 'avazu_feature_map')


# generate_valid_csv(test, feature_map)
# fill null with 0
# train.fillna(0, inplace=True)

# shuffle data
avazu = shuffle(avazu)

train_half = avazu.sample(frac=0.5, random_state=0, axis=0).reset_index(drop=True)
# storage to csv
train_half.to_csv('avazu_train_half.csv', index=None)
avazu.to_csv('avazu_train_full.csv', index=None)
