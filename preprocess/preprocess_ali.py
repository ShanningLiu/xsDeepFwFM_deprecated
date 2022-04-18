from sklearn.utils import shuffle
import pandas as pd
import warnings

from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter('ignore')

def mem_usage(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:  # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


raw_sample_df = pd.read_csv('ali_dataset/raw_sample.csv')
ad_feature_df = pd.read_csv('ali_dataset/ad_feature.csv')
user_profile_df = pd.read_csv('ali_dataset/user_profile.csv')

test_size_mb = raw_sample_df.memory_usage().sum() / 1024 / 1024
test_size_mb1 = ad_feature_df.memory_usage().sum() / 1024 / 1024
test_size_mb2 = user_profile_df.memory_usage().sum() / 1024 / 1024
print("raw_sample_df memory size: %.2f MB" % test_size_mb)
print("ad_feature_df memory size: %.2f MB" % test_size_mb1)
print("user_profile_df memory size: %.2f MB" % test_size_mb2)

raw_sample_df.info(memory_usage='deep')

optimized_gl = raw_sample_df.copy()

gl_int = raw_sample_df.select_dtypes(include=['int'])
converted_int = gl_int.apply(pd.to_numeric, downcast='unsigned')
optimized_gl[converted_int.columns] = converted_int

gl_obj = raw_sample_df.select_dtypes(include=['object']).copy()
converted_obj = pd.DataFrame()
for col in gl_obj.columns:
    num_unique_values = len(gl_obj[col].unique())
    num_total_values = len(gl_obj[col])
    if num_unique_values / num_total_values < 0.5:
        converted_obj.loc[:, col] = gl_obj[col].astype('category')
    else:
        converted_obj.loc[:, col] = gl_obj[col]
optimized_gl[converted_obj.columns] = converted_obj
print("Original Ad Feature dataframe:{0}".format(mem_usage(raw_sample_df)))
print("Memory Optimised Ad Feature dataframe:{0}".format(mem_usage(optimized_gl)))

raw_sample_df = optimized_gl.copy()
raw_sample_df_new = raw_sample_df.rename(columns={"user": "userid"})

ad_feature_df.info(memory_usage='deep')

optimized_g2 = ad_feature_df.copy()

g2_int = ad_feature_df.select_dtypes(include=['int'])
converted_int = g2_int.apply(pd.to_numeric, downcast='unsigned')
optimized_g2[converted_int.columns] = converted_int

g2_float = ad_feature_df.select_dtypes(include=['float'])
converted_float = g2_float.apply(pd.to_numeric, downcast='float')
optimized_g2[converted_float.columns] = converted_float

print("Original Ad Feature dataframe:{0}".format(mem_usage(ad_feature_df)))
print("Memory Optimised Ad Feature dataframe:{0}".format(mem_usage(optimized_g2)))

user_profile_df.info(memory_usage='deep')

optimized_g3 = user_profile_df.copy()

g3_int = user_profile_df.select_dtypes(include=['int'])
converted_int = g3_int.apply(pd.to_numeric, downcast='unsigned')
optimized_g3[converted_int.columns] = converted_int

g3_float = user_profile_df.select_dtypes(include=['float'])
converted_float = g3_float.apply(pd.to_numeric, downcast='float')
optimized_g3[converted_float.columns] = converted_float

print("Original User Feature dataframe:{0}".format(mem_usage(user_profile_df)))
print("Memory Optimised User Feature dataframe:{0}".format(mem_usage(optimized_g3)))

df1 = raw_sample_df_new.merge(optimized_g3, on="userid")
final_df = df1.merge(optimized_g2, on="adgroup_id")
final_df.head()

final_df['hist_cate_id'] = final_df['cate_id']
final_df['hist_adgroup_id'] = final_df['adgroup_id']

first_column = final_df.pop('clk')
final_df.insert(0, 'clk', first_column)

# = numerical features
dense_features = ['price', 'time_stamp']

final_df[dense_features] = final_df[dense_features].fillna(0, )

mms = MinMaxScaler(feature_range=(0, 1))
final_df[dense_features] = mms.fit_transform(final_df[dense_features])

for col in dense_features:
    tmp = final_df.pop(col)
    final_df.insert(1, col, tmp)

final_df.pop('nonclk')

# read raw dataset
ali_click = final_df
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
    for i in range(3, 20):
        for feature in feature_map[i]:
            if feature_map[i][feature] != 0:
                f_map.write(str(i) + ',' + str(feature) + ',' + str(feature_map[i][feature]) + '\n')
    return feature_map


def generate_valid_csv(inputs, feature_map):
    for i, col_map in enumerate(feature_map[2 + 1:]):
        inputs[inputs.columns[i +2 + 1]] = inputs[inputs.columns[i +2 + 1]].map(col_map)


#
# # Not the best way, follow xdeepfm
print('Count the frequency.')
ali_click.fillna(0, inplace=True)
freq_dict = cnt_freq_train(ali_click)

#
print('Generate the feature map and impute the training dataset.')
feature_map = generate_feature_map_and_train_csv(ali_click, freq_dict, 'ali_feature_map')


# shuffle data
ali_click = shuffle(ali_click)

# storage to csv
ali_click.to_csv('ali_full.csv', header=None, index=None)
