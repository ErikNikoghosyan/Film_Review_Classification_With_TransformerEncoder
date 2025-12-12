"""
custom_dst.py

CustomDataset — a lightweight PyTorch Dataset wrapper for binary text
classification that leverages Hugging Face tokenizers.

Responsibilities
- Tokenize and encode a collection of raw text strings into fixed-length
  tensors (input_ids, attention_mask) suitable for DataLoader batching.
- Store binary labels as float32 tensors for direct use with common loss
  functions (e.g., BCEWithLogitsLoss).
- Ensure a padding token exists for the tokenizer and perform deterministic
  padding/truncation to `max_length`.

Notes
- The current implementation performs eager (batch) tokenization in the
  constructor and keeps tensors in memory. For very large corpora, consider
  lazy tokenization in __getitem__ to reduce memory pressure.
- The class is intentionally lightweight and does not resize model embeddings
  if special tokens are added; resizing must be handled by the model owner.
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class CustomDataset(Dataset):
    """
    Custom dataset for text classification using a Hugging Face tokenizer.
    Handles attention masks and emojis correctly.
    """

    def __init__(self, texts, labels, tokenizer_name="vinai/bertweet-base", max_length=128, tokenizer=None):
        """
        Args:
            texts (list or pd.Series): List of text strings.
            labels (list or pd.Series): List of binary labels (0 or 1).
            tokenizer_name (str): Hugging Face tokenizer name.
            max_length (int): Maximum sequence length for padding/truncation.
        """
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
        self.tokenizer = tokenizer        
        self.max_length = max_length 
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})

        # Encode all texts at once
        self.encodings = self.tokenizer(
            texts.tolist() if hasattr(texts, "tolist") else texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        self.labels = torch.tensor(
            labels.values if hasattr(labels, "values") else labels,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

    def get_tokenizer_len(self):
        return self.tokenizer.vocab_size
