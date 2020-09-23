import pandas as pd
import torch
import transformers as ppb
import numpy as np
from bert_serving.client import BertClient

filepath = 'G:\\training_s.tsv'

def getBertEmbeddings(tokenized, pretrained_weights='distilbert-base-multilingual-cased'):
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, pretrained_weights)
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


chunksize = 10

names = ['text_tokens', 'hashtags', 'tweet_id', 'present_media', 'present_links', 'present_domains', 'tweet_type',
         'language', 'timestamp',
         'engaged_user_id', 'engagedfollower_count', 'engaged_following_count', 'engaged_verified',
         'engaged_account_creation_time',
         'engaging_user_id', 'engaging_follower_count', 'engaging_following_count', 'engaging_verified',
         'engaging_account_creation_time',
         'engagee_follows_engager', 'reply_engagement_timestamp', 'retweet_engagement_timestamp',
         'retweet_with_comment_engagement_timestamp', 'like_engagement_timestamp']

bc = BertClient()
print(bc.encode(['First do it', 'then do it right', 'then do it better']))

for enumerator, chunk in enumerate(pd.read_csv(filepath, sep='\x01', encoding='utf8', chunksize=chunksize, names=names)):
    text_tokens = chunk.text_tokens
    text_tokens = text_tokens.str.split('\t')

    print(bc.encode(text_tokens.tolist(), is_tokenized=True)[0][0])

    tokens = text_tokens.apply(lambda x: np.asarray(x, dtype=np.int).tolist())
    print(getBertEmbeddings(tokens)[0][0])
    break

bc.close()