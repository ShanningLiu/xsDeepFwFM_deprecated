import pandas as pd
import numpy as np
import mmh3
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import LeakyReLU
import tensorflow as tf
from sklearn.metrics import classification_report, balanced_accuracy_score, precision_recall_curve, auc, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RSEED = 42

pd.set_option('display.max_columns', None)

# filepath = "G:\\training.tsv"
filepath = "C:\\Users\\AndreasPeintner\\Downloads\\training.tsv"
# bert_filepath = "I:\\train_bert.tsv"

names = ['text_tokens', 'hashtags', 'tweet_id', 'present_media', 'present_links', 'present_domains', 'tweet_type',
         'language', 'timestamp',
         'engaged_user_id', 'engagedfollower_count', 'engaged_following_count', 'engaged_verified',
         'engaged_account_creation_time',
         'engaging_user_id', 'engaging_follower_count', 'engaging_following_count', 'engaging_verified',
         'engaging_account_creation_time',
         'engagee_follows_engager', 'reply_engagement_timestamp', 'retweet_engagement_timestamp',
         'retweet_with_comment_engagement_timestamp', 'like_engagement_timestamp']
numeric_features = ['engagedfollower_count', 'engaged_following_count', 'engaging_follower_count',
                    'engaging_following_count',
                    'timestamp', 'engaged_account_creation_time', 'engaging_account_creation_time']
categorical_features = ['language', 'tweet_type', 'engaged_verified', 'engaging_verified', 'engagee_follows_engager']
id_features = ['engaged_user_id', 'engaging_user_id',
               'tweet_id']
list_id_features = ['present_media', 'present_links', 'present_domains', 'hashtags']

CATEGORY = 'like_engagement_timestamp'
DATASIZE = 10 * 10 ** 3
BATCH_SIZE = 64
EPOCHS = 5


# https://stackoverflow.com/questions/39964424/how-to-get-a-dense-representation-of-one-hot-vectors
# test = np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
# print(tf.argmax(test, axis=1))

def splitPredictions(test_labels, test_probs):
    classes = np.unique(test_labels)

    labels = []
    preds = []
    for i, c in enumerate(classes):
        result_l = np.where(test_labels == c, 1, 0)
        result_p = [p[i] for p in test_probs]

        labels.append(result_l)
        preds.append(result_p)

    return labels, preds


def evaluate(test, test_labels):
    scores = model.evaluate(test, test_labels)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    y_test_pred = model.predict(test)
    #names = ['None', 'Reply', 'Retweet', 'Retweet with comment', 'Like']
    names = ['Reply', 'Retweet', 'Retweet with comment', 'Like']

    labels, preds = splitPredictions(np.argmax(test_labels, axis=1), y_test_pred)
    praucs, rces = 0, 0
    for i, l in enumerate(labels):
        prauc = compute_prauc(preds[i], l)
        rce = compute_rce(preds[i], l)
        print(names[i])
        print("\tPRAUC: ", prauc)
        print("\tRCE: ", rce)
        praucs = praucs + prauc
        rces = rces + rce

    print("\nAVG PRAUC: ", praucs / len(labels))
    print("AVG RC: ", rces / len(labels))


def compute_prauc(pred, gt):
    prec, recall, thresh = precision_recall_curve(gt, pred)
    prauc = auc(recall, prec)
    return prauc


def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive / float(len(gt))
    return ctr


def compute_rce(pred, gt):
    cross_entropy = log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy / strawman_cross_entropy) * 100.0


def cross_entropy(predictions, targets):
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions)) / N
    return ce


def get_binary_labels(chunk, className):
    chunk['label'] = 0
    chunk = chunk.assign(label=np.where(np.isnan(chunk[className]), chunk['label'], 1))
    labels = chunk.pop('label')

    return labels, chunk

def get_labels(chunk, ignoreNone=False):
    chunk['label'] = 4
    chunk = chunk.assign(label=np.where(np.isnan(chunk['reply_engagement_timestamp']), chunk['label'], 0))
    chunk = chunk.assign(label=np.where(np.isnan(chunk['retweet_engagement_timestamp']), chunk['label'], 1))
    chunk = chunk.assign(
        label=np.where(np.isnan(chunk['retweet_with_comment_engagement_timestamp']), chunk['label'], 2))
    chunk = chunk.assign(label=np.where(np.isnan(chunk['like_engagement_timestamp']), chunk['label'], 3))

    if ignoreNone:
        chunk.drop(chunk[chunk.label == 4].index, inplace=True)

    labels = chunk.pop('label')

    return labels, chunk


def get_model(input_dim=32, output_dim=4):
    model = Sequential()

    model.add(Dense(128, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Dense(output_dim, activation='sigmoid'))

    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss=tf.keras.losses.Huber(delta=100.0), metrics=["accuracy"])

    print(model.summary())
    return model


data = []
for chunk in pd.read_csv(filepath, sep='\x01', encoding='utf8', chunksize=DATASIZE, names=names,
                         converters={'hashtags': lambda x: x.split('\t'),
                                     'present_media': lambda x: x.split('\t'),
                                     'present_links': lambda x: x.split('\t'),
                                     'present_domains': lambda x: x.split('\t')}):
    data = chunk
    break

'''for bert_features in pd.read_csv(bert_filepath, sep=',', encoding='utf8', dtype=np.float64, chunksize=DATASIZE):
    data[['bert_' + str(i) for i in range(0, 768)]] = bert_features
    break'''

labels, data = get_labels(data, ignoreNone=True)

data.pop('reply_engagement_timestamp')
data.pop('retweet_engagement_timestamp')
data.pop('retweet_with_comment_engagement_timestamp')
data.pop('like_engagement_timestamp')

data['number_text_tokens'] = data['text_tokens'].apply(lambda x: str(x).count('\t') + 1)
data['number_hashtags'] = data['hashtags'].apply(lambda x: len(x) - 1)
data['present_media'] = data['present_media'].apply(lambda x: len(x) - 1)
data['present_links'] = data['present_links'].apply(lambda x: len(x) - 1)
data['present_domains'] = data['present_domains'].apply(lambda x: len(x) - 1)

data.pop('text_tokens')  # TODO
data.pop('hashtags')

for feature in numeric_features:
    data[feature] = pd.qcut(data[feature], q=50, labels=False)  # paper says 49 + 1
    one_hot = pd.get_dummies(data[feature], prefix=feature)
    data = pd.concat([data, one_hot], axis=1)
    data.pop(feature)

for feature in categorical_features:
    one_hot = pd.get_dummies(data[feature], prefix=feature)
    data = pd.concat([data, one_hot], axis=1)
    data.pop(feature)

for feature in id_features:
    data[feature] = data[feature].apply(lambda x: mmh3.hash(x, RSEED))
    data[feature] = pd.cut(data[feature], bins=200, labels=False)  # TODO number of bins?
    one_hot = pd.get_dummies(data[feature], prefix=feature)
    data = pd.concat([data, one_hot], axis=1)
    data.pop(feature)

# TODO split lists into columns https://stackoverflow.com/questions/35491274/pandas-split-column-of-lists-into-multiple-columns]}
'''
for feature in list_id_features:
    print(data[feature].tolist())
    data[feature] = data[feature].apply(lambda x: mmh3.hash(x, RSEED))
    data[feature] = pd.cut(data[feature], bins=200, labels=False) # TODO number of bins?
    one_hot = pd.get_dummies(data[feature], prefix=feature)
    data = pd.concat([data, one_hot], axis=1)
    data.pop(feature)'''

one_hot_labels = keras.utils.to_categorical(labels, num_classes=4)

train, test, train_labels, test_labels = train_test_split(data,
                                                          one_hot_labels,
                                                          stratify=one_hot_labels,
                                                          test_size=0.2,
                                                          random_state=42)

#print(data)
print(data.columns)

model = get_model(input_dim=len(data.columns), output_dim=4)

model.fit(train, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(test, test_labels), verbose=1)

evaluate(test, test_labels)


#TODO how to handle extern test data