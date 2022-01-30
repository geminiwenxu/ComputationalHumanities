import numpy as np
import torch
import torch.nn as nn


def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        re_targets = targets.reshape(len(targets), 1)
        re_targets = re_targets.to(torch.float32)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # print("train", outputs)
        preds = (outputs > 0.5).float()
        # print("train", preds)
        # print("train", re_targets)

        loss = loss_fn(outputs, re_targets)

        correct_predictions += torch.sum(preds == re_targets)
        # print("train", correct_predictions)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            re_targets = targets.reshape(len(targets), 1)
            re_targets = re_targets.to(torch.float32)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            preds = (outputs > 0.5).float()
            # print("eval", outputs)
            # print("eval", preds)
            # print("eval", re_targets)
            loss = loss_fn(outputs, re_targets)

            correct_predictions += torch.sum(preds == re_targets)
            # print("eval", correct_predictions)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)
