"""
Utilities for training, evaluating, and making single-sample predictions
with binary sentiment classification models in PyTorch.

This module provides:
- train: single-epoch training loop with optional batch-level logging.
- evaluate: evaluation loop that returns average loss and accuracy.
- predict_single: tokenize and predict a single text example.

Design goals:
- Clear type hints and docstrings.
- Robust device placement for inputs/targets.
- Safe conversion from model outputs to probabilities (works for logits or probs).
- Optional lightweight logging hook instead of hard prints.
"""

import torch 

def train(model, dataloader, optimizer, criterion, device, visualize=False):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    all_iters = len(dataloader)

    for iteration, batch in enumerate(dataloader, start=1):
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(inputs, attention_mask=attention_mask)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        preds = (outputs > 0.5).float()
        batch_accuracy = (preds == targets).float().mean()

        total_loss += loss.item()
        total_accuracy += batch_accuracy.item()

        if visualize:
            print(f"{iteration}/{all_iters} Train batch Loss: {loss.item():.4f}, "
                  f"Accuracy: {batch_accuracy:.4f}")

    avg_loss = total_loss / all_iters
    avg_accuracy = total_accuracy / all_iters

    return avg_loss, avg_accuracy


def evaluate(model, dataloader, criterion, device, visualize=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            targets = batch['labels'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()

            preds = (outputs > 0.5).float()

            correct = (preds == targets).float().sum().item()
            total_correct += correct
            total_samples += targets.numel()

            if visualize:
                batch_acc = correct / targets.numel()
                print(f"Eval Batch Loss: {loss.item():.4f}, Accuracy: {batch_acc:.4f}")

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy

def predict_single(model, dataset, text: str, device='cpu'):
    """
    Make a prediction on a single text example using the dataset tokenizer.

    Args:
        model: The trained PyTorch model
        dataset: Your CustomDataset instance (for tokenizer access)
        text (str): Input text
        device: 'cpu' or 'cuda'
    
    Returns:
        (sentiment_label, score)
    """
    model.eval()
    tokenizer = dataset.tokenizer  

    encoding = tokenizer(
        [text],  # batch of 1
        max_length=dataset.max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
        score = torch.sigmoid(logits).item()

    sentiment = "Positive" if score > 0.5 else "Negative"
    return sentiment, score
