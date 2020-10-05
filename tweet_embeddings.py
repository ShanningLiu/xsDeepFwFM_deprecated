import pandas as pd
import torch
import transformers as ppb
import numpy as np
from bert_serving.client import BertClient
from transformers import BertTokenizer

filepath = 'G:\\training_s.tsv'

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


chunksize = 128

names = ['text_tokens', 'hashtags', 'tweet_id', 'present_media', 'present_links', 'present_domains', 'tweet_type',
         'language', 'timestamp',
         'engaged_user_id', 'engagedfollower_count', 'engaged_following_count', 'engaged_verified',
         'engaged_account_creation_time',
         'engaging_user_id', 'engaging_follower_count', 'engaging_following_count', 'engaging_verified',
         'engaging_account_creation_time',
         'engagee_follows_engager', 'reply_engagement_timestamp', 'retweet_engagement_timestamp',
         'retweet_with_comment_engagement_timestamp', 'like_engagement_timestamp']
'''
print(getBertEmbeddings(["First do it"])[0][0])
print(getBertEmbeddingsfromTokens([[101, 12128, 10149, 10271, 102]])[0][0])
print(getBertEmbeddingsfromTokens([[101, 10422, 10149, 10271, 102]])[0][0])


tz = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
sent = "First do it"
print(tz.convert_tokens_to_ids(tz.tokenize(sent)))
sent = "first do it"
print(tz.convert_tokens_to_ids(tz.tokenize(sent)))'''

bc = BertClient(ip='35.230.16.12', port=6666)
print(bc.encode(['First do it'])[0][0])
print(bc.encode([['101', '12128', '10149', '10271', '102']], is_tokenized=True)[0][0])
print(bc.encode([['101', '10422', '10149', '10271', '102']], is_tokenized=True)[0][0])


for enumerator, chunk in enumerate(pd.read_csv(filepath, sep='\x01', encoding='utf8', chunksize=chunksize, names=names)):
    text_tokens = chunk.text_tokens
    text_tokens = text_tokens.str.split('\t')

    print(bc.encode(text_tokens.tolist(), is_tokenized=True)[0][0])
    '''
    tokens = text_tokens.apply(lambda x: np.asarray(x, dtype=np.int).tolist())
    print(getBertEmbeddings(tokens)[0][0])
    break'''

bc.close()