import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torchsummary # type: ignore
from torch.utils.data import DataLoader, Dataset # type: ignore
from torch.utils.tensorboard import SummaryWriter # type: ignore

import os
import json
import numpy as np # type: ignore
from tqdm import tqdm # type: ignore
import pickle
from sklearn.metrics import accuracy_score # type: ignore

from tokenizers import Tokenizer # type: ignore
from model import MathDocClassifier
from data.data_processing import MathDataset, MathDatasetCleaner


processed_test_file_path = 'data/test-data-processed.pkl'

if os.path.exists(processed_test_file_path):
    with open(processed_test_file_path, 'rb') as file:
        test_data, test_formulas = pickle.load(file)
        print(f"Loading test data from Cache.")
else:
    test_data, test_formulas = MathDatasetCleaner("data/example-test-data.jsonl.gz", label_file_path="data/example-test-results.json").parse()
    with open(processed_test_file_path, 'wb') as file:
        pickle.dump((test_data, test_formulas), file)        
    print(f"The file training data is saved at: {processed_test_file_path}.")

tokenizer = Tokenizer.from_file("data/mathml_tokenizer.json")

test_dataset = MathDataset(documents=test_data, tokenizer=tokenizer, max_formulas=10, max_length=150)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=True)

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
    "n_classes": test_dataset.n_classes,
    "vocab_size": test_dataset.vocab_size,
    "optimizer": "AdamW",
    "epochs": 150
}

model = MathDocClassifier(hparams)

# Load the pretrained weights
pretrained_weights = torch.load('ckpts/best_model.pth')

# Load these weights into the model
model.load_state_dict(pretrained_weights)

model.eval()
all_preds = []
all_doc_ids = []

device = "cuda" if torch.cuda.is_available() else "cpu"

with torch.no_grad():
    with tqdm(test_loader, desc=f"Validation") as val_pbar:
        for batch in val_pbar:
            formulas = batch["input"].to(device)
            doc_ids = batch["doc_id"].squeeze().to(device)
            
            outputs = model(formulas)
            
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_doc_ids.extend(doc_ids.cpu())

        result = {
            f"paper{doc_id:04d}": test_dataset.get_label(class_label) for doc_id, class_label in zip(all_doc_ids, all_preds)
        }

        with open("data/my_test_result.json", "w") as f:
            json.dump(dict(result), f, indent=4)