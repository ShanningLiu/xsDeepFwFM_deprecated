from sklearn.utils import shuffle
import pandas as pd
import warnings

from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter('ignore')


final_df = pd.read_csv('train_ali_full.csv', index_col=None, low_memory=False)

first_column = final_df.pop('clk')
final_df.insert(0, 'clk', first_column)

# = numerical features
dense_features = ["price_x","price_y","brand_influence","bc_influence","buy_pv_brand","cart_pv_brand","fav_pv_brand","buy_pv_bc","cart_pv_bc","fav_pv_bc"]

final_df[dense_features] = final_df[dense_features].fillna(0, )

mms = MinMaxScaler(feature_range=(0, 1))
final_df[dense_features] = mms.fit_transform(final_df[dense_features])

for col in dense_features:
    tmp = final_df.pop(col)
    final_df.insert(1, col, tmp)

final_df.pop('nonclk')

print(final_df.head())
print(final_df.shape)

# read raw dataset
# ali_click = shuffle(final_df)
# ali_click(0, inplace=True)

# final_df.to_csv('test_after_idx.csv', header=None, index=None)

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
    for i, col_map in enumerate(feature_map[10 + 1:]):
        inputs[inputs.columns[i + 10 + 1]] = inputs[inputs.columns[i + 10 + 1]].map(col_map)

    # write feature_map file
    f_map = open(file_feature_map, 'w')
    for i in range(11, 41):
        for feature in feature_map[i]:
            if feature_map[i][feature] != 0:
                f_map.write(str(i) + ',' + str(feature) + ',' + str(feature_map[i][feature]) + '\n')
    return feature_map


def generate_valid_csv(inputs, feature_map):
    for i, col_map in enumerate(feature_map[10 + 1:]):
        inputs[inputs.columns[i + 10 + 1]] = inputs[inputs.columns[i + 10 + 1]].map(col_map)

# ali_click = pd.read_csv('test_after_idx.csv', header=None, index_col=None, low_memory=False)
ali_click = final_df

ali_click = shuffle(ali_click)
print(ali_click.head())
#

# ali_click = pd.read_csv('ali_after_idx.csv', header=None, index_col=None, low_memory=False)
# # Not the best way, follow xdeepfm
print('Count the frequency.')
ali_click.fillna(0, inplace=True)
freq_dict = cnt_freq_train(ali_click)

#
print('Generate the feature map and impute the training dataset.')
feature_map = generate_feature_map_and_train_csv(ali_click, freq_dict, 'ali_feature_test_map')

# generate_valid_csv(test, feature_map)


# shuffle data
# ali_click = shuffle(ali_click)
# ali_click.fillna(0, inplace=True)
# storage to csv
# ali_click.to_csv('ali_train.csv', header=None, index=None)
train_half = ali_click.sample(frac=0.5, random_state=0, axis=0).reset_index(drop=True)
# storage to csv
train_half.to_csv('ali_train_half.csv', index=None)

# shuffle data
# test = shuffle(test)
# test.fillna(0, inplace=True)
# # storage to csv
# test.to_csv('ali_test.csv', header=None, index=None)