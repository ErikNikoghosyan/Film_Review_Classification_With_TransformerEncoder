"""
main.py

Entrypoint and orchestration utilities for the Film Reviews Classification project.

This module provides clear, production-friendly functions to:
- prepare data (delegates to src.data_code.data_process.process_save_data),
- construct datasets and dataloaders (CustomDataset),
- instantiate a lightweight Transformer encoder (SimpleTransformerEncoder),
- train and evaluate models, and
- run interactive single-sample inference.

Design goals
- Explicit type hints and small, well-documented public functions.
- Robust filesystem handling using pathlib.Path and deterministic behavior.
- Minimal side effects: directories and files are created only when needed.
- Light-weight logging via the standard logging module (configurable by caller).
"""

import os
import torch
import pandas as pd

from torch.utils.data import DataLoader
from src.data_code.data_process import process_save_data
from src.data_code.custom_dst import CustomDataset
from src.models.simple_transformer_encoder import SimpleTransformerEncoder
from src.utils import train, evaluate, predict_single
from transformers import AutoTokenizer



def main_train_evaluate(
        # Define parameters
        # For training 
        raw_data_path: str = 'data/raw/netflix_reviews.csv',
        train_data_path: str = 'data/processed/train_data.csv',
        test_data_path: str = 'data/processed/test_data.csv',
        best_model_folder_path: str = 'data/models/best_model.pth',
        batch_size: int = 32,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        # For model
        model_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        fc_dropout: float = 0.1,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        device: str = 'cuda',
        # Whether to save the best model
        save_best_model: bool = True,
        # For visulization
        visualize: bool = True
):

    # Process and save data
    try: 
        csv_train_data = pd.read_csv(train_data_path)
        csv_test_data = pd.read_csv(test_data_path)
    except FileNotFoundError:
        print("Processed data not found, processing raw data...")
        os.makedirs('data/processed', exist_ok=True)
        process_save_data(raw_data_path, "data/processed")
        csv_train_data = pd.read_csv("data/processed/train_data.csv")
        csv_test_data = pd.read_csv("data/processed/test_data.csv")
        print("Data processing complete.")
        print("As train and test data paths weren't found, raw data has been processed and saved to data/processed/. Be careful to set the correct paths if needed.")
    
    
    # Create datasets
    train_dataset = CustomDataset(csv_train_data["content"], csv_train_data["sentiment"], max_length=max_seq_length)
    test_dataset = CustomDataset(csv_test_data["content"], csv_test_data["sentiment"], max_length=max_seq_length)
    tokenizer = train_dataset.tokenizer
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Device availibility
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = SimpleTransformerEncoder(
        input_dim=train_dataset.get_tokenizer_len(),
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        fc_dropout=fc_dropout,
        dropout=dropout,
        max_seq_length=max_seq_length,
        device=device
    ).to(device)

    # Criterion and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    if save_best_model:
        best_loss_file = "data/models/best_loss.txt"
        best_model_path = f"{best_model_folder_path}"+"/best_model.pth"
    for epoch in range(num_epochs):
        # Handel the file where we will save best model's accuracy
        if save_best_model and os.path.exists(best_loss_file):
            with open(best_loss_file, "r") as f:
                try:
                    best_loss = float(f.read().strip())
                except:
                    best_loss = float('inf')
        else:            
            best_loss = float('inf')

        # One epoch training
        loss, accuracy = train(model, 
                     train_loader, 
                     optimizer=optimizer, 
                     criterion=criterion, 
                     device=device, 
                     visualize=visualize, 
                     )
        
        # Saving best model and the best loss
        if save_best_model and loss < best_loss:
            print("NEW BEST MODEL !!!")
            best_loss = loss
            #Saving model dict
            try:
                torch.save(model.state_dict(), best_model_path)
                tokenizer.save_pretrained("data/models/tokenizer/")            
            except:
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                os.makedirs(os.path.dirname("data/models/tokenizer/"), exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                tokenizer.save_pretrained("data/models/tokenizer/")

            # Saving model best loss
            with open(best_loss_file, "w") as f:
                f.write(f"{best_loss:.6f}")
            print(f"Model is saved in {best_model_path}")
        
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {loss:.4f}, Training Accuracy: {accuracy:.4f}")
    
    # Evaluate the model
    test_loss, test_accuracy = evaluate(model, test_loader, criterion=criterion, device=device, visualize=visualize)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")


    print("Model initialized successfully.")


def run_the_best(
        # Path for saved model 
        model_path: str = 'data/models/best_model.pth',
        # For training / data
        raw_data_path: str = 'data/raw/netflix_reviews.csv',
        train_data_path: str = 'data/processed/train_data.csv',
        test_data_path: str = 'data/processed/test_data.csv',
        batch_size: int = 32,
        # Model hyperparams
        model_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        device: str = 'cuda',
        # Behavior
        eval_train=True,
        eval_test=True,
        self_input=False,
        tokenizer_local_path: str = "data/models/tokenizer/",
        **unnecesary_params
):

    try:
        csv_train_data = pd.read_csv(train_data_path)
        csv_test_data = pd.read_csv(test_data_path)
    except FileNotFoundError:
        print("Processed data not found, processing raw data...")
        os.makedirs('data/processed', exist_ok=True)
        process_save_data(raw_data_path, "data/processed")
        csv_train_data = pd.read_csv("data/processed/train_data.csv")
        csv_test_data = pd.read_csv("data/processed/test_data.csv")
        print("Data processing complete.")

    # --- Load tokenizer (prefer local saved tokenizer) ---
    tokenizer = None
    if os.path.isdir(tokenizer_local_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_local_path, use_fast=False)
            print(f"Loaded tokenizer from local path: {tokenizer_local_path}")
        except Exception as e:
            print(f"Failed to load local tokenizer at {tokenizer_local_path}: {e}. Falling back to 'vinai/bertweet-base'.")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
        print("Loaded tokenizer from 'vinai/bertweet-base'")

    # --- Create datasets & loaders (ensure dataset stores max_length) ---
    train_dataset = CustomDataset(csv_train_data["content"], csv_train_data["sentiment"],
                                 max_length=max_seq_length, tokenizer=tokenizer)
    test_dataset = CustomDataset(csv_test_data["content"], csv_test_data["sentiment"],
                                max_length=max_seq_length, tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Build model ---
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SimpleTransformerEncoder(
        input_dim=train_dataset.get_tokenizer_len(),
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        max_seq_length=max_seq_length,
        device=device
    )

    # --- Robustly load state dict ---
    map_loc = device
    if not os.path.exists(model_path):
        print(f"Warning: model file {model_path} not found. Running evaluation with randomly initialized model.")
    else:
        try:
            saved = torch.load(model_path, map_location=map_loc)
            # saved may be a state_dict or dict with extra keys (like optimizer)
            if isinstance(saved, dict) and all(isinstance(k, str) for k in saved.keys()):
                # If it's a checkpoint with keys like 'model_state_dict'
                if 'model_state_dict' in saved and isinstance(saved['model_state_dict'], dict):
                    state_dict = saved['model_state_dict']
                else:
                    state_dict = saved
            else:
                state_dict = saved

            # Try strict load first
            try:
                model.load_state_dict(state_dict)
                print(f"Loaded model state_dict (strict=True) from {model_path}")
            except Exception as e_strict:
                print(f"Strict load failed: {e_strict}")
                # Try partial load by matching keys and shapes
                model_sd = model.state_dict()
                matched = {}
                for k, v in state_dict.items():
                    if k in model_sd and v.shape == model_sd[k].shape:
                        matched[k] = v
                if len(matched) == 0:
                    print("No matching keys found between saved state and model. Can't partially load.")
                else:
                    model_sd.update(matched)
                    model.load_state_dict(model_sd)
                    print(f"Partially loaded {len(matched)} tensors into the model (matched by name & shape).")
        except Exception as e:
            print(f"Failed to load model file {model_path}: {e}")

    # Move model to device (ensure on correct device after partial load)
    model.to(device)
    model.eval()

    # --- Criterion ---
    criterion = torch.nn.BCEWithLogitsLoss()

    # --- Evaluate ---
    if eval_train:
        train_loss, train_accuracy = evaluate(model, train_loader, criterion=criterion, device=device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    if eval_test:
        test_loss, test_accuracy = evaluate(model, test_loader, criterion=criterion, device=device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # --- Interactive prediction ---
    if self_input:
        print("\nEnter your review (or type 'exit'):\n")
        while True:
            review = input("Review: ").strip()
            if review.lower() == "exit":
                break
            sentiment, score = predict_single(model, train_dataset, review, device)
            print(f"Prediction → Sentiment: {sentiment} (score={score:.4f})\n")


if __name__ == '__main__':
    HYPER_PARAMS = {
        'raw_data_path': 'data/raw/netflix_reviews.csv',
        'train_data_path': 'data/processed/train_data.csv',
        'test_data_path': 'data/processed/test_data.csv',
        'best_model_folder_path': 'data/models',
        'batch_size': 16,
        'num_epochs': 10,
        'learning_rate': 5e-5,
        'model_dim': 256,
        'num_heads': 32,
        'num_layers': 8,
        'fc_dropout':0.5,
        'dropout': 0.0,
        'max_seq_length': 256,
        'device': 'cuda',
        'save_best_model': True,
        'visualize': True
    }
    # main_train_evaluate(**HYPER_PARAMS)
    run_the_best(**HYPER_PARAMS, eval_test=False, eval_train=False, self_input=True)