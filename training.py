import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import numpy as np
import os

from models.dti_model import DTIModel

def load_dataset(path):
    data_list = torch.load(path, weights_only=False)
    return data_list

def collate_fn(batch):

    from torch_geometric.data import Batch
    pyg_batch = Batch.from_data_list(batch)
    protein = torch.stack([d.protein for d in batch])
    y = torch.stack([d.y for d in batch]).view(-1)
    return pyg_batch, protein, y

def compute_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        raise ValueError("All predictions or targets are NaN.")
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    for batch, protein, y in tqdm(loader, desc="Train", leave=False):
        batch, protein, y = batch.to(device), protein.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            pred = model(batch, protein)
            loss = nn.functional.mse_loss(pred, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for batch, protein, y in tqdm(loader, desc="Eval", leave=False):
        batch, protein, y = batch.to(device), protein.to(device), y.to(device)
        pred = model(batch, protein)
        y_true.append(y)
        y_pred.append(pred)
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    return compute_metrics(y_true, y_pred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='processed/bindingdb_train.pt')
    parser.add_argument('--test_path', type=str, default='processed/bindingdb_test.pt')
    parser.add_argument('--davis_path', type=str, default='processed/davis.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='best_dti_model.pt')
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load datasets
    train_set = load_dataset(args.train_path)
    test_set = load_dataset(args.test_path)
    davis_set = load_dataset(args.davis_path)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    davis_loader = DataLoader(davis_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)


    sample = train_set[0]
    node_dim = sample.x.size(1)
    edge_dim = sample.edge_attr.size(1) if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 0

    model = DTIModel(node_dim=node_dim, edge_dim=edge_dim)
    model = model.to(args.device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, dynamic=True)
   

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    scaler = torch.amp.GradScaler('cuda') if args.use_fp16 and args.device.startswith('cuda') else None

    best_rmse = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, args.device)
        test_rmse, test_mae, test_r2 = evaluate(model, test_loader, args.device)
        train_rmse, train_mae, train_r2 = evaluate(model, train_loader, args.device)
        scheduler.step(test_rmse)

        print(f"Epoch {epoch:03d} | Train RMSE: {train_rmse:.4f} MAE: {train_mae:.4f} R2: {train_r2:.4f} | "
              f"Test RMSE: {test_rmse:.4f} MAE: {test_mae:.4f} R2: {test_r2:.4f}")

        if test_rmse + args.min_delta < best_rmse:
            best_rmse = test_rmse
            patience_counter = 0
            torch.save(model.state_dict(), args.save_path)
            print(f"Best model saved at epoch {epoch} (Test RMSE: {test_rmse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    # Load best model and evaluate on Davis
    model.load_state_dict(torch.load(args.save_path, map_location=args.device))
    davis_rmse, davis_mae, davis_r2 = evaluate(model, davis_loader, args.device)
    print(f"Davis Evaluation | RMSE: {davis_rmse:.4f} MAE: {davis_mae:.4f} R2: {davis_r2:.4f}")

if __name__ == "__main__":
    main()
