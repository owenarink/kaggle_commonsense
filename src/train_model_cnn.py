
from src.models.cnn_text import CNN_TFIDF
from src.features.tfidf import get_tfidf
from src.processing import preprocess
from torch.utils.data import DataLoader, TensorDataset
from src.features.tfidf import get_tfidf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import csv
import os
import torch
import torch.nn as nn
from src.eval import evaluate_model
from tqdm import tqdm


def load_prep_data():
    x_train_raw = pd.read_csv('data/train_data.csv')
    x_test_raw = pd.read_csv('data/test_data.csv') 
    y_train_raw = pd.read_csv('data/train_answers.csv')
    
    x_train_preprocessed, y_train_preprocessed = preprocess(x_train_raw, x_test_raw, y_train_raw)
    x_train, x_val, y_train, y_val = get_tfidf(x_train_preprocessed, y_train_preprocessed)

    return x_train, x_val, y_train, y_val

def train(x_train, x_val, y_train, y_val):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).long())
    val_dataset = TensorDataset(torch.tensor(x_val).float(), torch.tensor(y_val).long())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # model
    model = CNN_TFIDF(n_inputs=x_train.shape[1], n_classes = 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_crit = nn.CrossEntropyLoss()

    
    epochs = 10

    # total progress bar across all batches in all epochs
    total_steps = epochs * len(train_loader)
    total_pbar = tqdm(total=total_steps, desc="Total", unit="batch")

    for epoch in range(epochs):
        model.train()

        # per-epoch progress bar
        epoch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False)

        for batch_x, batch_y in epoch_pbar:
            outputs = model(batch_x)
            loss = loss_crit(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update both progress bars
            epoch_pbar.set_postfix(loss=f"{loss.item():.4f}")
            total_pbar.update(1)

        epoch_pbar.close()

        # accuracy prints after each epoch
        train_acc = evaluate_model(model, train_loader)
        val_acc = evaluate_model(model, val_loader)
        print(f"epoch [{epoch+1}/{epochs}] loss={loss.item():.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    total_pbar.close()

    # ---- ADD THIS: save model after training ----
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/cnn_text.pt")
    print("Saved model to checkpoints/mlp_tfidf.pt")

if __name__ == "__main__":
    x_train, x_val, y_train, y_val = load_prep_data()   
    print('------------------- TRAIN MODEL ----------------')
    print('train_model, after preprocess and get_tfidf: x_train', x_train)
    print('train_model, after preprocess and get_tfidf: x_val', x_val)
    print('train_model, after preprocess and get_tfidf: y_train', y_train)
    print('train_model, after preprocess and get_tfidf: y_val', y_train)
 
    train(x_train, x_val, y_train, y_val)


