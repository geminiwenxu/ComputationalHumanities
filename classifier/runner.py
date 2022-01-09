from classifier.bert_model import BertBinaryClassifier
from classifier.prepare_data import prepare_data
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from transformers import BertTokenizer
import json
import yaml
from pkg_resources import resource_filename

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
BATCH_SIZE = 1
EPOCHS = 3
bert_clf = BertBinaryClassifier()
bert_clf.to(device)
optimizer = torch.optim.Adam(bert_clf.parameters(), lr=3e-6)
train_dataloader, train_data, dev_dataloader, dev_data, dev_y = prepare_data(BATCH_SIZE)
class_names = ['human written', 'machine written']


def main():
    # training model
    for epoch_num in range(EPOCHS):
        bert_clf.train()
        train_loss = 0
        for step_num, batch_data in enumerate(train_dataloader):
            token_ids, masks, labels = tuple(t for t in batch_data)
            token_ids = token_ids.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            probas = bert_clf(token_ids, masks)
            loss_func = nn.BCELoss()
            batch_loss = loss_func(probas, labels).to(device)
            train_loss += batch_loss.item()
            bert_clf.zero_grad()
            batch_loss.backward()
            optimizer.step()
            print('Epoch: ', epoch_num + 1)
            print(
                "\r" + "{0}/{1} loss: {2} ".format(step_num, len(train_data) / BATCH_SIZE, train_loss / (step_num + 1)))

    # evaluate model
    bert_clf.eval()
    bert_predicted = []
    all_logits = []
    with torch.no_grad():
        for step_num, batch_data in enumerate(dev_dataloader):
            token_ids, masks, labels = tuple(t for t in batch_data)
            token_ids = token_ids.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            logits = bert_clf(token_ids, masks)
            loss_func = nn.BCELoss().to(device)
            loss = loss_func(logits, labels)
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
