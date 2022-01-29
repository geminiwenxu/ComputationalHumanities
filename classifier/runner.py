from classifier_2.bert_model import BertBinaryClassifier
from classifier_2.prepare_data import create_data_loader
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

import json
import yaml
from pkg_resources import resource_filename
from transformers import BertTokenizerFast
import pandas as pd
import numpy as np


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
EPOCHS = 20
bert_clf = BertBinaryClassifier()
bert_clf.to(device)
optimizer = torch.optim.Adam(bert_clf.parameters(), lr=0.01)

class_names = ['human written', 'machine written']
MAX_LEN = 512

config = get_config('/../config/config.yaml')
train_path = resource_filename(__name__, config['train']['path'])
dev_path = resource_filename(__name__, config['dev']['path'])
test_path = resource_filename(__name__, config['test']['path'])
BATCH_SIZE = 16
df_train = pd.read_json(train_path)

df_dev = pd.read_json(dev_path)

dev_data = [{'text': text, 'label': type_data} for text in list(df_dev['text']) for type_data in
            list(df_dev['label_id'])]
dev_labels = df_dev['label_id']
dev_y = np.array(dev_labels) == 1

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
dev_data_loader = create_data_loader(df_dev, tokenizer, MAX_LEN, BATCH_SIZE)
train_df = pd.read_json(train_path)
train_data = [{'text': text, 'label': type_data} for text in list(train_df['text']) for type_data in
              list(train_df['label_id'])]


def main():
    # training model
    for epoch_num in range(EPOCHS):
        bert_clf.train()
        train_loss = 0
        for step_num, d in enumerate(train_data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].unsqueeze(1)
            targets = targets.float()
            targets = targets.to(device)
            probas = bert_clf(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            loss_func = nn.BCELoss()
            batch_loss = loss_func(probas, targets).to(device)
            train_loss += batch_loss.item()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(bert_clf.parameters(), max_norm=1.0)
            optimizer.step()
            bert_clf.zero_grad()
            print('Epoch: ', epoch_num + 1)
            print(
                "\r" + "{0}/{1} loss: {2} ".format(step_num, len(train_data) / BATCH_SIZE, train_loss / (step_num + 1)))

    # evaluate model
    bert_clf.eval()
    bert_predicted = []
    all_logits = []
    with torch.no_grad():
        for step_num, d in enumerate(dev_data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].unsqueeze(1)
            targets = targets.float()
            targets = targets.to(device)
            logits = bert_clf(input_ids, attention_mask)
            loss_func = nn.BCELoss().to(device)
            loss = loss_func(logits, targets)
            numpy_logits = logits.cpu().detach().numpy()
            bert_predicted += list(numpy_logits[:, 0] > 0.5)
            all_logits += list(numpy_logits[:, 0])
    print(classification_report(dev_y, bert_predicted))

    # test model
    def get_config(path):
        with open(resource_filename(__name__, path), 'r') as stream:
            conf = yaml.safe_load(stream)
        return conf

    config = get_config('/../config/config.yaml')
    test_path = resource_filename(__name__, config['test']['path'])
    result = []
    f = open(test_path)
    data = json.load(f)
    for i in data:
        dict = {}
        review_text = i['text']
        prediction_id = i['index']
        index = i['index']
        encoded_review = tokenizer.encode_plus(
            review_text,
            max_length=512,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        token_ids = encoded_review['input_ids'].to(device)
        attention_mask = encoded_review['attention_mask'].to(device)

        output = bert_clf(token_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)
        prediction = class_names[prediction]

        dict['prediction_id'] = prediction_id
        dict['prediction'] = prediction
        dict['index'] = index
        result.append(dict)
    with open("submission.json", "w", encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
