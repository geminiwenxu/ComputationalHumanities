import pandas as pd
import numpy as np
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
import json

if __name__ == '__main__':
    train_df = pd.read_json(
        '/recognition_dataset-labeled/test.json')
    # dev_df = pd.read_json(
    #     '/Users/wenxu/PycharmProjects/ComputationalHumanities/recognition_dataset-labeled/dataset_dev.json')

    train_data = [{'text': text, 'label': type_data} for text in list(train_df['text']) for type_data in
                  list(train_df['label_id'])]

    # dev_data = [{'text': text, 'label': type_data} for text in list(dev_df['text']) for type_data in
    #             list(dev_df['label_id'])]

    train_texts, train_labels = list(zip(*map(lambda d: (d['text'], d['label']), train_data)))
    # dev_texts, dev_labels = list(zip(*map(lambda d: (d['text'], d['type']), dev_data)))


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:511], train_texts))
    # dev_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:511], dev_texts))

    train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))
    # dev_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, dev_tokens))
    train_tokens_ids = pad_sequences(train_tokens_ids, maxlen=512, truncating="post", padding="post", dtype="int")
    # test_tokens_ids = pad_sequences(dev_tokens_ids, maxlen=512, truncating="post", padding="post", dtype="int")

    train_y = np.array(train_labels) == 1
    # dev_y = np.array(dev_labels) == 1




