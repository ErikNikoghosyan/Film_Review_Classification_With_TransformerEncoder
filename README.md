# Film Reviews Classification

A lightweight PyTorch-based binary sentiment classification system for film reviews, specifically designed for Netflix reviews. This project leverages a Transformer encoder architecture with Hugging Face tokenizers to classify reviews as positive or negative.

## Project Overview

This project implements an end-to-end pipeline for:
- **Data Processing**: Raw CSV ingestion, cleaning, and stratified train/test splits
- **Model Training**: Full training loop with checkpointing and best-model selection
- **Evaluation**: Comprehensive metrics on train/test sets
- **Inference**: Interactive single-sample prediction interface

The implementation emphasizes production-friendly design with:
- Clear type hints and docstrings
- Robust error handling and logging
- Efficient memory usage via batch tokenization
- Deterministic behavior with configurable random states

## Directory Structure

```
Film_reviews_classification/
├── main.py                        # Entry point and orchestration
├── data/
│   ├── raw/                       # Raw CSV(s)
│   ├── processed/                 # Train/test CSVs produced by preprocessing
│   └── models/                    # Saved model state and tokenizer
├── src/
│   ├── data_code/
│   │   ├── data_process.py        # Data preparation utilities
│   │   └── custom_dst.py          # PyTorch Dataset wrapper
│   ├── models/
│   │   └── simple_transformer_encoder.py  # Model backbone
│   └── utils.py                   # Train / evaluate / predict helpers
└── notebooks/
    └── test.ipynb                 # Data exploration and quick checks
```

## Requirements

- Python 3.8+
- PyTorch (compatible with your CUDA / CPU setup)
- transformers
- pandas
- scikit-learn

Recommended install (adjust versions to match your environment):

```bash
# Clone or download the project
cd Film_reviews_classification

# Install dependencies
pip install torch transformers pandas scikit-learn
```

Optionally add plotting libraries for notebooks:

```bash
pip install matplotlib seaborn
```

You may also create a `requirements.txt` with pinned versions for reproducibility.

## Data

Expect a CSV under `data/raw/` with at least the columns:

- `content` — review text (string)
- `score` — numeric rating (1–5)

Processing performed by `process_save_data()` (see `src/data_code/data_process.py`):

- Drops rows with missing `content` or `score`
- Converts `score` to integer and keeps only clearly negative and positive examples (configurable; by default negative scores → 1–2, positive → 4–5)
- Maps scores to a binary `sentiment` column: `0` = negative, `1` = positive
- Optionally filters by review length (default < 150 characters)
- Produces a stratified 80/20 train/test split saved to `data/processed/`

Example:

```python
from src.data_code.data_process import process_save_data

process_save_data(
    input_file_path="data/raw/netflix_reviews.csv",
    output_folder_path="data/processed",
)
```

## Model

`SimpleTransformerEncoder` (see `src/models/simple_transformer_encoder.py`) implements:

- **Token Embedding**: Converts input token IDs to embeddings
- **Positional Encoding**: Sinusoidal positional encodings to preserve sequence order
- **Transformer Layers**: Stacked encoder layers with multi-head attention
- **Pooling**: Mean pooling over the sequence dimension
- **Classification Head**: Fully connected layers with dropout for binary classification

**Architecture Configuration:**
```
Input Tokens → Embedding + Positional Encoding → Transformer Encoder (6 layers, 8 heads)
→ Mean Pooling → FC Layers (256 → 128 → 64 → 32 → 1) → Logits
```

**Output**: Raw logits suitable for `BCEWithLogitsLoss`

## Quickstart — Training

From Python (programmatic):

```python
from main import main_train_evaluate

main_train_evaluate(
    raw_data_path="data/raw/netflix_reviews.csv",
    train_data_path="data/processed/train_data.csv",
    test_data_path="data/processed/test_data.csv",
    models_dir="data/models",
    batch_size=16,
    num_epochs=10,
    learning_rate=5e-5,
    model_dim=256,
    num_heads=8,
    num_layers=4,
    max_seq_length=256,
    device="cuda",  # or "cpu"
    save_best_model=True,
    visualize=False,
)
```

Or run the module directly to use the default `HYPER_PARAMS` configured in `main.py`:

```bash
python main.py
```

## Quickstart — Evaluation & Inference

```python
from main import run_the_best

run_the_best(
    model_path="data/models/best_model.pth",
    train_data_path="data/processed/train_data.csv",
    test_data_path="data/processed/test_data.csv",
    batch_size=32,
    device="cuda",
    eval_train=True,
    eval_test=True,
    self_input=True,  # interactive loop for single-sample input
)
```

For single-sample prediction programmatically:

```python
from src.utils import predict_single
from src.data_code.custom_dst import CustomDataset
import torch

# Load dataset and model
dataset = CustomDataset(texts, labels)
model = torch.load("data/models/best_model.pth")
model.eval()

# Predict
label, score = predict_single(model, dataset, "This movie is amazing!", device="cuda")
print(f"Sentiment: {label}, Score: {score:.4f}")
```

## Core Modules

### `main.py`
Entry point and orchestration module providing:
- `main_train_evaluate()`: Full training pipeline with data processing, model training, and evaluation
- `run_the_best()`: Load trained model, evaluate on datasets, and run interactive predictions

### `src/data_code/data_process.py`
Data preparation utilities:
- `process_save_data()`: Loads raw CSV, filters by score, removes missing values, performs train/test split

### `src/data_code/custom_dst.py`
PyTorch Dataset wrapper:
- Handles tokenization using Hugging Face tokenizers
- Returns batches as dicts with `input_ids`, `attention_mask`, and `labels`
- Eager batch tokenization for efficient DataLoader integration

### `src/models/simple_transformer_encoder.py`
Neural network model:
- `SimpleTransformerEncoder`: Transformer encoder with positional encodings and classification head
- Returns raw logits for binary classification

### `src/utils.py`
Training and evaluation utilities:
- `train()`: Single-epoch training loop with optional visualization
- `evaluate()`: Evaluation loop returning loss and accuracy
- `predict_single()`: Tokenize text and return sentiment prediction

## Hyperparameters

Key configurable parameters in `main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 16 | Batch size for training/evaluation |
| `num_epochs` | 10 | Number of training epochs |
| `learning_rate` | 5e-5 | Adam optimizer learning rate |
| `model_dim` | 256 | Transformer hidden dimension |
| `num_heads` | 8 | Number of attention heads |
| `num_layers` | 4 | Number of Transformer encoder layers |
| `max_seq_length` | 256 | Maximum token sequence length |
| `dropout` | 0.1 | Dropout rate in Transformer |
| `fc_dropout` | 0.5 | Dropout rate in classification head |
| `device` | "cuda" | Training device (cuda/cpu) |

## Training Loop

1. **Data Loading**: Processed train/test CSVs loaded (created if missing)
2. **Dataset Creation**: `CustomDataset` tokenizes and encodes all texts
3. **Model Initialization**: `SimpleTransformerEncoder` instantiated and moved to device
4. **Training**: For each epoch:
   - Compute loss with `BCEWithLogitsLoss`
   - Backpropagate and update weights
   - Save best model if training loss improves
5. **Evaluation**: Final test set evaluation with loss and accuracy metrics

## Logging

The project uses Python's `logging` module (INFO level by default). Adjust verbosity in `main.py`:

```python
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
```

## Tests / Notebooks

Check `notebooks/test.ipynb` for initial data exploration and small smoke tests such as verifying the data split and average review length.

## License

Provided as-is for educational and research purposes. Adapt or extend as you wish.

## Author

Erik Nikoghosyan

---

If you'd like, I can also:

- add a `requirements.txt` with pinned versions,
- implement the masked mean pooling in `SimpleTransformerEncoder`, or
- add a small smoke-test script verifying preprocessing + a single forward pass.

## GIFT — Pretrained model bundle

If you want to distribute a ready-to-run model so users can skip training, prepare a single compressed bundle containing the model state and tokenizer. We refer to this artifact as the "pretrained model bundle" or "gift".

Bundle contents (recommended layout):

```
model_bundle.zip
├── best_model.pth              # PyTorch state_dict (model.state_dict())
├── metadata.json               # Optional: model config, training hyperparams
└── tokenizer/                  # Hugging Face tokenizer files (tokenizer.json, vocab, config...)
```

Packaging instructions:

1. Place `best_model.pth` (the saved state_dict) and the tokenizer directory in a temporary folder, e.g. `model_bundle/`.
2. (Optional) Add a small `metadata.json` describing model architecture, `model_dim`, `max_seq_length`, and the tokenizer id used during training.
3. Compress the folder into `model_bundle.zip`.

Usage (no training required):

1. Unzip the bundle into the repository (recommended target: `data/models/<bundle_name>/`). Example:

```bash
unzip model_bundle.zip -d data/models/my_bundle
```

2. Load the model with the existing helpers. Example (programmatic):

```python
from main import run_the_best

run_the_best(
    model_path="data/models/my_bundle/best_model.pth",
    tokenizer_local_path="data/models/my_bundle/tokenizer",
    eval_test=True,
    self_input=True,
)
```

Notes and best practices:

- Save only the model `state_dict` (not the full model object) to keep the bundle framework-agnostic. Use `torch.save(model.state_dict(), path)` when exporting.
- Include `metadata.json` with the bundle so users know which `model_dim`, `num_layers`, and `max_seq_length` to use when instantiating `SimpleTransformerEncoder`.
- The bundle should include the tokenizer directory produced by `tokenizer.save_pretrained(...)` so tokenization matches training.

Git tracking and `.gitignore`:

By default large binary artifacts are often ignored. To ensure your bundle can be committed and is not accidentally ignored, add an explicit whitelist entry to `.gitignore`. Example lines to add to the repository `.gitignore`:

```
# Allow committing pretrained model bundles placed under data/models/
!/data/models/*.zip
!/data/models/**/*.zip
```

Place these whitelist lines after any `data/models/` or similar ignore rules so they take precedence. Note: `.gitignore` prevents files from being tracked — it does not delete files from disk. The whitelist above ensures Git will consider the `.zip` for tracking even if parent folders are otherwise ignored.

Security & size considerations:

- Model bundles can be large; consider hosting them on an external release server (GitHub Releases, S3) and include a small README with download instructions if you prefer not to store large binaries in the repo history.
- Do not include sensitive data or credentials inside the bundle.

If you want, I can:

- add the recommended whitelist lines to the repository `.gitignore`, and
- add a short `scripts/save_bundle.sh` that packages the `best_model.pth`, tokenizer, and `metadata.json` into a `.zip` ready for distribution.
