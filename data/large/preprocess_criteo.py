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

def random_split(inputs, output1, valid, test):
    fout1 = open(output1, 'w')
    fout2 = open(valid, 'w')
    fout3 = open(test, 'w')

    num_lines = sum(1 for line in open(inputs))

    for line_number, line in enumerate(open(inputs)):
        # all except last day for training set
        if line_number < num_lines - (num_lines // 7):
            fout1.write(line)
        else:
            # randomly split last day into validation and test set
            if random.uniform(0, 1) < 0.5:
                fout2.write(line)
            else:
                fout3.write(line)
    fout1.close()
    fout2.close()
    fout3.close()

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

def generate_valid_csv(inputs, valid_csv, feature_map):
    fout = open(valid_csv, 'w')
    for idx, line in enumerate(open(inputs)):
        line = line.replace('\n', '').split('\t')
        output_line = [line[0]]
        for i in range(1, 40):
            if i < 14:
                #line[i] = project_numeric(line[i])
                line[i] = scale(line[i])
                output_line.append(line[i])
            elif line[i] in feature_map[i]:
                output_line.append(feature_map[i][line[i]])
            else:
                output_line.append('0')
        output_line = ','.join(output_line)
        fout.write(output_line + '\n')

file = 'G:\\dac\\train_ss.txt'
#file = 'C:\\Users\\AndreasPeintner\\Documents\\dac\\train_s.txt'

# no test data with labels online available
print('Split the orignal dataset into train and valid dataset.')
random_split(file, 'train1.txt', 'valid.txt', 'test.txt')

# Not the best way, follow xdeepfm
print('Count the frequency.')
freq_dict = cnt_freq_train('train1.txt')

print('Generate the feature map and impute the training dataset.')
feature_map = generate_feature_map_and_train_csv('train1.txt', 'criteo_train.csv', 'criteo_feature_map', freq_dict, threshold=4)

print('Impute the valid dataset.')
generate_valid_csv('valid.txt', 'criteo_valid.csv', feature_map)
generate_valid_csv('test.txt', 'criteo_test.csv', feature_map)

print('Delete unnecessary files')
os.remove('valid.txt')
os.remove('test.txt')
os.remove('train1.txt')
