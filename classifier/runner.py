from classifier.bert_model import BertBinaryClassifier
from classifier.prepare_data import create_data_loader
import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.metrics import classification_report
from transformers import get_linear_schedule_with_warmup
import json
import yaml
from pkg_resources import resource_filename
from transformers import BertTokenizerFast
import pandas as pd
from classifier.train import train_epoch, eval_model
from classifier.prediction import get_predictions
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ['machine-generated', 'human-written']
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)

model = BertBinaryClassifier(len(class_names))
model.to(device)

MAX_LEN = 160
BATCH_SIZE = 16


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../config/config.yaml')
train_path = resource_filename(__name__, config['train']['path'])
dev_path = resource_filename(__name__, config['dev']['path'])
test_path = resource_filename(__name__, config['test']['path'])

df_train = pd.read_json(train_path)
df_dev = pd.read_json(dev_path)

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
dev_data_loader = create_data_loader(df_dev, tokenizer, MAX_LEN, BATCH_SIZE)

EPOCHS = 2
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)


def main():
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(EPOCHS):

        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            dev_data_loader,
            loss_fn,
            device,
            len(df_dev)
        )

        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')

    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig('epoch.png', bbox_inches='tight')

    # evaluate on unseen data (not sure the dev dataset is the validation or to be tested?)
    # dev_acc, _ = eval_model(
    #     model,
    #     dev_data_loader,
    #     loss_fn,
    #     device,
    #     len(df_dev)
    # )
    #
    # print(dev_acc.item())
    # y_pred, y_pred_probs, y_dev = get_predictions(
    #     model,
    #     dev_data_loader
    # )
    # print(classification_report(y_dev, y_pred, target_names=class_names))

    # test model
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

        output = model(token_ids, attention_mask)
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
