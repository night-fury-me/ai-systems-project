import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torchsummary # type: ignore
from torch.utils.data import DataLoader, Dataset # type: ignore
from torch.utils.tensorboard import SummaryWriter # type: ignore
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, LinearLR, CosineAnnealingLR # type: ignore

import os
import numpy as np # type: ignore
from tqdm import tqdm # type: ignore
import pickle
import mlflow # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

from tokenizers import Tokenizer # type: ignore
from model import MathDocClassifier
from data.data_processing import MathDataset, MathDatasetCleaner

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("New-Math-Document-Classification")

def configure_optimizer(model, train_loader, hparams):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=hparams["training"]["lr"],
        weight_decay=hparams["training"]["weight_decay"]
    )
    
    if hparams["scheduling"]["scheduler"] == "cosine":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hparams["scheduling"]["cycle_length"],
            eta_min=hparams["scheduling"]["min_lr"]
        )
    elif hparams["scheduling"]["scheduler"] == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=hparams["training"]["lr"],
            total_steps=hparams["epochs"] * len(train_loader),
            pct_start=0.3
        )
    elif hparams["scheduling"]["scheduler"] == "linear":
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=hparams["scheduling"]["warmup_epochs"] * len(train_loader)
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, CosineAnnealingLR(optimizer, T_max=hparams["epochs"])],
            milestones=[hparams["scheduling"]["warmup_epochs"] * len(train_loader)]
        )
    
    return optimizer, scheduler

def train_model(model, train_loader, val_loader, hparams, device="cpu"):
    model.to(device)
    epochs = hparams["epochs"]
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"])
    
    optimizer, scheduler = configure_optimizer(model, train_loader, hparams)

    best_acc = 0

    with tqdm(total=epochs, desc="Training", unit="epoch", position=0, ascii='->',
             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            all_preds = []
            all_labels = []
            
            with tqdm(train_loader, desc=f"Epoch [{epoch:3d}]", position=1, leave=False, ascii='->',
                     bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as batch_pbar:

                for b_idx, batch in enumerate(batch_pbar):
                    formulas = batch["input"].to(device)  # (batch_size, max_formulas, max_length)
                    labels   = batch["label"].squeeze().to(device)
                    
                    # print(f"{formulas.shape=}")

                    optimizer.zero_grad()
                    
                    outputs = model(formulas)
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    t_acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
                    
                    batch_pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{t_acc:.2f}%"
                    })
            
            scheduler.step()
            
            # Log epoch training metrics
            train_loss /= len(train_loader)
            train_acc = accuracy_score(all_labels, all_preds)

            # Update progress bar with training metrics
            pbar.set_postfix_str(
                f"t_acc: {train_acc:.4f}, v_acc: ??"
            )

            val_acc, val_loss = evaluate_model(model, val_loader, device)

            lr = optimizer.param_groups[0]['lr']

            # MLFLOW - Log metrics for each epoch
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc" : train_acc,
                "val_loss"  : val_loss,
                "val_acc"   : val_acc,
                "lr"        : lr
            }, step = epoch)

            # Update progress bar with both metrics
            pbar.set_postfix_str(
                f"t_acc: {train_acc:.4f}, v_acc: {val_acc:.4f}"
            )
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "ckpts/best_model.pth")

            # Advance epoch progress bar
            pbar.update(1)

    print(f"Best Validation Accuracy: {best_acc:.4f}")

def evaluate_model(model, loader, device="cpu"):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        with tqdm(loader, desc=f"Validation", position=1, leave=False, ascii='->',
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as val_pbar:
            for batch in val_pbar:
                formulas = batch["input"].to(device)
                labels = batch["label"].squeeze().to(device)
                
                outputs = model(formulas)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                v_acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
                val_pbar.set_postfix({
                    'val_acc': f"{v_acc:.2f}%"
                })
    
    return accuracy_score(all_labels, all_preds), total_loss/len(loader)


if __name__ == "__main__":

    processed_training_file_path = 'data/training-data-processed.pkl'
    processed_test_file_path = 'data/test-data-processed.pkl'

    if os.path.exists(processed_training_file_path):
        with open(processed_training_file_path, 'rb') as file:
            train_data, train_formulas = pickle.load(file)
            print(f"Loading training data from Cache.")

    else:
        train_data, train_formulas, labels = MathDatasetCleaner("data/training-data.jsonl.gz").parse()
        with open(processed_training_file_path, 'wb') as file:
            pickle.dump((train_data, train_formulas), file)        
        print(f"The file training data is saved at: {processed_training_file_path}.")

    if os.path.exists(processed_test_file_path):
        with open(processed_test_file_path, 'rb') as file:
            test_data, test_formulas = pickle.load(file)
            print(f"Loading test data from Cache.")
    else:
        test_data, test_formulas, labels = MathDatasetCleaner("data/example-test-data.jsonl.gz", label_file_path="data/example-test-results.json").parse()
        with open(processed_test_file_path, 'wb') as file:
            pickle.dump((test_data, test_formulas), file)        
        print(f"The file training data is saved at: {processed_test_file_path}.")
    
    with mlflow.start_run():
        tokenizer = Tokenizer.from_file("data/mathml_tokenizer.json")
        
        # Create datasets
        train_dataset = MathDataset(documents=train_data, tokenizer=tokenizer, max_formulas=10, max_length=350)
        train_loader  = DataLoader(train_dataset, batch_size=128, shuffle=True)

        test_dataset = MathDataset(documents=test_data, tokenizer=tokenizer, max_formulas=10, max_length=350)
        test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)

        print(f"Vocab Size = {train_dataset.vocab_size}")

        hparams = {
            "FormulaTransformer": {
                "d_model": 256,         
                "n_heads": 8,           # 256 / 32 = 8 heads
                "ff_dim": 512,          # 2 x d_model
                "num_layers": 3,        
                "dropout": 0.4, 
            },
            "DocumentTransformer": {
                "d_model": 256,
                "n_heads": 8,
                "ff_dim": 768,
                "num_layers": 4,        
                "dropout": 0.3,         
            },
            "training": {
                "lr": 3e-4,             
                "weight_decay": 0.1,    
                "batch_size": 256,      
                "label_smoothing": 0.1, 
                "grad_clip": 1.0
            },
            "classifier" : {
                "dropout" : 0.4
            },
            "scheduling": {
                "scheduler": "cosine",
                "warmup_epochs": 5,
                "min_lr": 1e-6,
                "cycle_length": 20      
            },
            "early_stopping": {
                "patience": 15,
                "min_delta": 0.001
            },
            "data": {
                "augmentation_prob": 0.5,  
                "mask_prob": 0.15          
            },
            "formula_max_length" : 350,
            "n_classes": train_dataset.n_classes,
            "vocab_size": train_dataset.vocab_size,
            "optimizer": "AdamW",
            "epochs": 150
        }

        mlflow.log_params(hparams)

        # Initialize model
        model = MathDocClassifier(hparams)

        # Log model architecture
        with open("data/model_architecture.txt", "w") as f:
            f.write(str(model))
            
        mlflow.log_artifact("data/model_architecture.txt")

        # # Load the pretrained weights
        # pretrained_weights = torch.load('ckpts/last_trained_best_model.pth')

        # # Load these weights into the model
        # model.load_state_dict(pretrained_weights)

        mlflow.log_artifact('ckpts/last_trained_best_model.pth', 'ckpts')

        # Train
        print("Started training the model...")
        train_model(model, train_loader, test_loader, hparams, device="cuda")
        # Log final model
        mlflow.pytorch.log_model(model, "model")

        # Log model
        mlflow.log_artifact('ckpts/best_model.pth', 'ckpts')

        # Log dataset info
        mlflow.log_artifact("data/training-data-processed.pkl", "data")
        mlflow.log_artifact("data/test-data-processed.pkl", "data")
        mlflow.log_artifact("data/mathml_tokenizer.json", "data")
        
        # Log training code snapshot
        mlflow.log_artifact("model.py")
        mlflow.log_artifact("train.py")
        mlflow.log_artifact("generate_result.py")
        mlflow.log_artifact("data/data_processing.py", "data")
        mlflow.log_artifact("data/math_symbol_extractor.py", "data") 

        # Add custom tags
        mlflow.set_tags({
            "task": "math-document-classification",
            "framework": "pytorch"
        })