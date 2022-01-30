import torch.nn as nn
from transformers import AutoModel


class BertBinaryClassifier(nn.Module):
    def __init__(self):
        super(BertBinaryClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask, return_dict=False)
        drop_output = self.drop(pooled_output)
        linearoutput = self.out(drop_output)
        prob = self.sigmoid(linearoutput)
        return prob
