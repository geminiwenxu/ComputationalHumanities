import pandas as pd
import numpy as np
from transformers import BertTokenizer
import torch
from keras.preprocessing.sequence import pad_sequences
import yaml
from pkg_resources import resource_filename


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../config/config.yaml')
train_path = resource_filename(__name__, config['train']['path'])
dev_path = resource_filename(__name__, config['dev']['path'])
test_path = resource_filename(__name__, config['test']['path'])


def prepare_data(BATCH_SIZE):
    train_df = pd.read_json(train_path)
    dev_df = pd.read_json(dev_path)

    train_data = [{'text': text, 'label': type_data} for text in list(train_df['text']) for type_data in
                  list(train_df['label_id'])]
    dev_data = [{'text': text, 'label': type_data} for text in list(dev_df['text']) for type_data in
                list(dev_df['label_id'])]

    train_texts, train_labels = list(zip(*map(lambda d: (d['text'], d['label']), train_data)))
    dev_texts, dev_labels = list(zip(*map(lambda d: (d['text'], d['label']), dev_data)))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:511], train_texts))
    dev_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:511], dev_texts))

    train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))
    dev_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, dev_tokens))

    train_tokens_ids = pad_sequences(train_tokens_ids, maxlen=512, truncating="post", padding="post", dtype="int")
    dev_tokens_ids = pad_sequences(dev_tokens_ids, maxlen=512, truncating="post", padding="post", dtype="int")

    train_y = np.array(train_labels) == 1
    dev_y = np.array(dev_labels) == 1

    train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]
    dev_masks = [[float(i > 0) for i in ii] for ii in dev_tokens_ids]

    train_masks_tensor = torch.tensor(train_masks)
    dev_masks_tensor = torch.tensor(dev_masks)

    train_tokens_tensor = torch.tensor(train_tokens_ids)
    train_y_tensor = torch.tensor(train_y.reshape(-1, 1)).float()
    dev_tokens_tensor = torch.tensor(dev_tokens_ids)
    dev_y_tensor = torch.tensor(dev_y.reshape(-1, 1)).float()

    train_dataset = torch.utils.data.TensorDataset(train_tokens_tensor, train_masks_tensor, train_y_tensor)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE,
                                                   num_workers=4)

    dev_dataset = torch.utils.data.TensorDataset(dev_tokens_tensor, dev_masks_tensor, dev_y_tensor)
    dev_sampler = torch.utils.data.SequentialSampler(dev_dataset)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, sampler=dev_sampler, batch_size=BATCH_SIZE, num_workers=4)

    return train_dataloader, train_data, dev_dataloader, dev_data, dev_y
