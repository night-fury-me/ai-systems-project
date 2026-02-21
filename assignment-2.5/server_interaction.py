
"""
To use this implementation, you simply have to implement `get_classifications` such that it returns classifications.
You can then let your agent compete on the server by calling

    python3 server_interaction.py path/to/your/config.json
"""
import json
import logging

import requests
import time

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

def get_classifications(request):
    # TODO: Return better classifications

    # print(len(request[0]))
    test_data, train_formulas = MathDatasetCleaner().parse_server_data(request)
    tokenizer = Tokenizer.from_file("data/mathml_tokenizer.json")

    test_dataset = MathDataset(documents=test_data, tokenizer=tokenizer, max_formulas=10, max_length=150)
    test_loader  = DataLoader(test_dataset, batch_size=min(256, len(request)), shuffle=False)

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
        with tqdm(test_loader, desc=f"Testing", position=1, leave=False, ascii='->',
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as test_pbar:
            for batch in test_pbar:
                formulas = batch["input"].to(device)                
                outputs = model(formulas)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())

        result = [test_dataset.get_label(c_label) for c_label in all_preds]
    assert len(result) == len(request), "Result len is not same as request"
    return result


def run(config_file, action_function, parallel_runs=True):
    logger = logging.getLogger(__name__)

    with open(config_file, 'r') as fp:
        config = json.load(fp)

    actions = []

    with tqdm(total=201, desc="Request", unit="request", position=0, ascii='->',
             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as req_pbar:

        for request_number in range(201):    # 100 runs are enough for full evaluation. Running much more puts unnecessary strain on the server's database.
            logger.info(f'Iteration {request_number} (sending {len(actions)} actions)')
            # send request
            response = requests.put(f'{config["url"]}/act/{config["env"]}', json={
                'agent': config['agent'],
                'pwd': config['pwd'],
                'actions': actions,
                'single_request': not parallel_runs,
            })
            if response.status_code == 200:
                response_json = response.json()
                for error in response_json['errors']:
                    logger.error(f'Error message from server: {error}')
                for message in response_json['messages']:
                    logger.info(f'Message from server: {message}')

                action_requests = response_json['action-requests']
                if not action_requests:
                    logger.info('The server has no new action requests - waiting for 1 second.')
                    time.sleep(1)  # wait a moment to avoid overloading the server and then try again
                # get actions for next request
                actions = []
                with tqdm(action_requests, desc=f"Action Request", position=1, leave=False, ascii='->',
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as action_pbar:
                    for action_request in action_pbar:
                        actions.append({'run': action_request['run'], 'action': action_function(action_request['percept'])})
            elif response.status_code == 503:
                logger.warning('Server is busy - retrying in 3 seconds')
                time.sleep(3)  # server is busy - wait a moment and then try again
            else:
                # other errors (e.g. authentication problems) do not benefit from a retry
                logger.error(f'Status code {response.status_code}. Stopping.')
                break

            req_pbar.update(1)
    print('Done - 100 runs are enough for full evaluation')


if __name__ == '__main__':
    import sys
    run(sys.argv[1], get_classifications)
