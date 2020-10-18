import pandas as pd
import torch
import transformers as ppb
import numpy as np
from bert_serving.client import BertClient
from transformers import BertTokenizer

input = 'G:\\training_final.tsv'
output = 'G:\\training_final_bert.csv'

open(output, 'w').close()

def getBertEmbeddings(sentences, pretrained_weights='bert-base-multilingual-cased'):
    model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, pretrained_weights)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    encoder = lambda x: tokenizer.encode(x, add_special_tokens=True)
    tokenized = list(map(encoder, sentences))
    print(tokenized)

    max_len = 0
    for i in tokenized:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
    input_ids = torch.tensor(np.array(padded)).type(torch.LongTensor)

    with torch.no_grad():
        last_hidden_states = model(input_ids)

    vectors = last_hidden_states[0][:, 0, :].numpy()
    return vectors

def getBertEmbeddingsfromTokens(tokenized, pretrained_weights='bert-base-multilingual-cased'):
    model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    max_len = 0
    for i in tokenized:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
    input_ids = torch.tensor(np.array(padded)).type(torch.LongTensor)

    with torch.no_grad():
        last_hidden_states = model(input_ids)

    vectors = last_hidden_states[0][:, 0, :].numpy()
    return vectors


chunksize = 512
skip_chunks = 1650

names = ['text_tokens', 'hashtags', 'tweet_id', 'present_media', 'present_links', 'present_domains', 'tweet_type',
         'language', 'timestamp',
         'engaged_user_id', 'engagedfollower_count', 'engaged_following_count', 'engaged_verified',
         'engaged_account_creation_time',
         'engaging_user_id', 'engaging_follower_count', 'engaging_following_count', 'engaging_verified',
         'engaging_account_creation_time',
         'engagee_follows_engager', 'reply_engagement_timestamp', 'retweet_engagement_timestamp',
         'retweet_with_comment_engagement_timestamp', 'like_engagement_timestamp']
bc = BertClient(ip='35.247.77.188', port=6666)
#bc = BertClient()

for enumerator, chunk in enumerate(pd.read_csv(input, sep='\x01', encoding='utf8', chunksize=chunksize, names=names)):
    #if enumerator < skip_chunks:
        #continue

    text_tokens = chunk.text_tokens
    text_tokens = text_tokens.str.split('\t')

    #print(bc.encode(text_tokens.tolist(), is_tokenized=True)[0][0])
    features = bc.encode(text_tokens.tolist(), is_tokenized=True)

    #tokens = text_tokens.apply(lambda x: np.asarray(x, dtype=np.int).tolist())
    #features = getBertEmbeddingsfromTokens([tokens.iloc[511]])
    #print(features[0][0])

    df = pd.DataFrame(features, columns=['bert_' + str(i) for i in range(0, features.shape[1])])
    df.to_csv(output, sep=',', encoding='utf8', index=False, mode='a', header=False)
    print(enumerator)

#bc.close()