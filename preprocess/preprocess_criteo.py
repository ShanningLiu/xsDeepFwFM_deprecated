"""
Preprocess Criteo raw data

First, you can download the raw dataset dac.tar.gz using the link below
http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/

Unzip the raw data:
>> tar xvzf dac.tar.gz

Criteo data has 13 numerical fields and 26 category fields.

In the training dataset

Step 1, split data into 6 days for training set and last day into validation and test set

Step 2, cnt_freq_train: count the frequency of features in each field;
        ignore data that has less than 40 columns (it consists of 47% of the whole dataset)

Step 3, ignore_long_tail: set the long-tail features with frequency less than a threshold as 0; \
        generate the feature-map index, the columns are: field index, unique feature, mapped index

In the valid dataset map the known feature to existing index and set the unknown as 0.

"""

import math
import os
import random

random.seed(0)

# https://github.com/WayneDW/AutoInt/blob/master/Dataprocess/Criteo/scale.py
def scale(x):
    if x == '':
        return '0'
    elif float(x) > 2:
        return str(int(math.log(float(x))**2))  # log transformation to normalize numerical features
    else:
        return str(int(float(x)))

def cnt_freq_train(inputs):
    count_freq = []
    for i in range(40):
        count_freq.append({})
    for idx, line in enumerate(open(inputs)):
        line = line.replace('\n', '').split('\t')
        if idx % 1000000 == 0 and idx > 0:
            print(idx)
        for i in range(1, 40):
            if i < 14:
                #line[i] = project_numeric(line[i])
                line[i] = scale(line[i])
            if line[i] not in count_freq[i]:
                count_freq[i][line[i]] = 0
            count_freq[i][line[i]] += 1
    return count_freq


def generate_feature_map_and_train_csv(inputs, train_csv, file_feature_map, freq_dict, threshold=4):
    feature_map = []
    for i in range(40):
        feature_map.append({})
    fout = open(train_csv, 'w')
    for idx, line in enumerate(open(inputs)):
        line = line.replace('\n', '').split('\t')
        if idx % 1000000 == 0 and idx > 0:
            print(idx)
        output_line = [line[0]]
        for i in range(1, 40):
            # map numerical features
            if i < 14:
                #line[i] = project_numeric(line[i])
                line[i] = scale(line[i])
                output_line.append(line[i])
            # handle categorical features
            elif freq_dict[i][line[i]] < threshold:
                output_line.append('0')
            elif line[i] in feature_map[i]:
                output_line.append(feature_map[i][line[i]])
            else:
                output_line.append(str(len(feature_map[i]) + 1))
                feature_map[i][line[i]] = str(len(feature_map[i]) + 1)
        output_line = ','.join(output_line)
        fout.write(output_line + '\n')

    # write feature_map file
    f_map = open(file_feature_map, 'w')
    for i in range(1, 40):
        #only_one_zero_index = True
        for feature in feature_map[i]:
            #if feature_map[i][feature] == '0' and only_one_zero_index == False:
            #    continue
            f_map.write(str(i) + ',' + feature + ',' + feature_map[i][feature] + '\n')
            #if only_one_zero_index == True and feature_map[i][feature] == '0':
            #    only_one_zero_index = False
    return feature_map

file = '/Users/liushanning/Desktop/2516/xsDeepFwFM_deprecated/data/large/dac/train.txt'

# Not the best way, follow xdeepfm
print('Count the frequency.')
freq_dict = cnt_freq_train(file)

print('Generate the feature map and impute the training dataset.')
feature_map = generate_feature_map_and_train_csv(file, 'criteo_train.csv', 'criteo_feature_map', freq_dict, threshold=4)

