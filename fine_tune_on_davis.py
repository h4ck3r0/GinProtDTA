import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import os

from models.dti_model import DTIModel
from protein_encoder import ProteinBERTEncoder

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
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return rmse, mae, r2

def train_one_epoch(model, loader, optimizer, scaler, device, use_fp16=False):
    model.train()
    total_loss = 0
    for batch, protein, y in tqdm(loader, desc="Train", leave=False):
        batch, protein, y = batch.to(device), protein.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu', enabled=use_fp16):
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
def evaluate(model, loader, device, use_fp16=False):
    model.eval()
    y_true, y_pred = [], []
    for batch, protein, y in tqdm(loader, desc="Eval", leave=False):
        batch, protein, y = batch.to(device), protein.to(device), y.to(device)
        with torch.amp.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu', enabled=use_fp16):
            pred = model(batch, protein)
        y_true.append(y)
        y_pred.append(pred)
    y_true = torch.cat([t.view(-1).float() for t in y_true if isinstance(t, torch.Tensor) and t.numel() > 0])
    y_pred = torch.cat([t.view(-1).float() for t in y_pred if isinstance(t, torch.Tensor) and t.numel() > 0])
    if y_true.numel() == 0 or y_pred.numel() == 0:
        return float('nan'), float('nan'), float('nan')
    if torch.isnan(y_true).all() or torch.isnan(y_pred).all():
        return float('nan'), float('nan'), float('nan')
    return compute_metrics(y_true, y_pred)

def freeze_layers(model, freeze_until=0):
    # Optionally freeze early layers for faster fine-tuning
    count = 0
    for name, param in model.named_parameters():
        if count < freeze_until:
            param.requires_grad = False
        count += 1

def gradual_unfreeze(model, total_layers, current_epoch, total_epochs):
    # Gradually unfreeze layers as training progresses
    layers_to_unfreeze = int(total_layers * current_epoch / total_epochs)
    count = 0
    for name, param in model.named_parameters():
        if count < layers_to_unfreeze:
            param.requires_grad = True
        count += 1

def ensemble_predict(models, loader, device, use_fp16=False):
    preds = []
    for model in models:
        model.eval()
        fold_preds = []
        for batch, protein, _ in loader:
            batch, protein = batch.to(device), protein.to(device)
            with torch.amp.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu', enabled=use_fp16):
                pred = model(batch, protein)
            fold_preds.append(pred.detach().cpu())
        preds.append(torch.cat(fold_preds))
    mean_preds = torch.stack(preds).mean(dim=0)
    return mean_preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--davis_path', type=str, default='processed/davis.pt')
    parser.add_argument('--pretrained_path', type=str, default='best_dti_model.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=1e-5)
    parser.add_argument('--save_path', type=str, default='best_dti_model_finetuned.pt')
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gnn_hidden_dim', type=int, default=256)
    parser.add_argument('--gnn_layers', type=int, default=4)
    parser.add_argument('--mlp_hidden_dim', type=int, default=256)
    parser.add_argument('--mlp_layers', type=int, default=3)
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'max', 'sum', 'attention', 'set2set'])
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--freeze_layers', type=int, default=0)
    parser.add_argument('--ensemble_top_n', type=int, default=3)
    # Hyperparameter grid
    parser.add_argument('--lr_grid', type=str, default='1e-5,5e-5,1e-4')
    parser.add_argument('--batch_grid', type=str, default='32,64')
    parser.add_argument('--dropout_grid', type=str, default='0.1,0.2,0.3')
    args = parser.parse_args()

    davis_set = load_dataset(args.davis_path)
    sample = davis_set[0]
    node_dim = sample.x.size(1)
    edge_dim = sample.edge_attr.size(1) if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 0

    # Prepare hyperparameter grid
    lr_grid = [float(x) for x in args.lr_grid.split(',')]
    batch_grid = [int(x) for x in args.batch_grid.split(',')]
    dropout_grid = [float(x) for x in args.dropout_grid.split(',')]

    best_overall_r2 = -float('inf')
    best_hparams = None
    best_models = []

    for lr in lr_grid:
        for batch_size in batch_grid:
            for dropout in dropout_grid:
                print(f"\nGrid Search: lr={lr}, batch_size={batch_size}, dropout={dropout}")
                kf = KFold(n_splits=args.kfold, shuffle=True, random_state=42)
                all_metrics = []
                fold_models = []
                for fold, (train_idx, val_idx) in enumerate(kf.split(davis_set)):
                    print(f"\nFold {fold+1}/{args.kfold}")
                    train_subset = Subset(davis_set, train_idx)
                    val_subset = Subset(davis_set, val_idx)
                    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
                    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

                    model = DTIModel(
                        node_dim=node_dim,
                        edge_dim=edge_dim,
                        gnn_hidden_dim=args.gnn_hidden_dim,
                        gnn_layers=args.gnn_layers,
                        mlp_hidden_dim=args.mlp_hidden_dim,
                        mlp_layers=args.mlp_layers,
                        gnn_pooling=args.pooling,
                        gnn_dropout=dropout,
                        mlp_dropout=dropout,
                        use_attention_pooling=True,
                        use_set2set_pooling=True
                    )
                    model = model.to(args.device)
                    model.load_state_dict(torch.load(args.pretrained_path, map_location=args.device))
                    if args.freeze_layers > 0:
                        freeze_layers(model, args.freeze_layers)
                    if args.compile and hasattr(torch, "compile"):
                        model = torch.compile(model, dynamic=True)

                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=args.weight_decay)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
                    scaler = torch.amp.GradScaler('cuda') if args.use_fp16 and args.device.startswith('cuda') else None

                    best_acc = -float('inf')
                    patience_counter = 0
                    total_layers = sum(1 for _ in model.named_parameters())

                    for epoch in range(1, args.epochs + 1):
                        # Gradual unfreezing
                        gradual_unfreeze(model, total_layers, epoch, args.epochs)
                        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, args.device, use_fp16=args.use_fp16)
                        val_rmse, val_mae, val_r2 = evaluate(model, val_loader, args.device, use_fp16=args.use_fp16)
                        val_acc = val_r2
                        scheduler.step(val_rmse)

                        print(f"Epoch {epoch:03d} | Val RMSE: {val_rmse:.4f} MAE: {val_mae:.4f} R2: {val_r2:.4f}")

                        if val_acc > best_acc + args.min_delta:
                            best_acc = val_acc
                            patience_counter = 0
                            torch.save(model.state_dict(), f"{args.save_path}_grid{lr}_{batch_size}_{dropout}_fold{fold+1}.pt")
                            print(f"Best model saved at epoch {epoch} (Val R2: {val_r2:.4f})")
                        else:
                            patience_counter += 1
                            if patience_counter >= args.patience:
                                print("Early stopping triggered.")
                                break

                    # Load best model for this fold and evaluate
                    model.load_state_dict(torch.load(f"{args.save_path}_grid{lr}_{batch_size}_{dropout}_fold{fold+1}.pt", map_location=args.device))
                    val_rmse, val_mae, val_r2 = evaluate(model, val_loader, args.device, use_fp16=args.use_fp16)
                    print(f"Fold {fold+1} Final | RMSE: {val_rmse:.4f} MAE: {val_mae:.4f} R2: {val_r2:.4f}")
                    all_metrics.append((val_rmse, val_mae, val_r2))
                    fold_models.append(model)

                avg_rmse = np.mean([m[0] for m in all_metrics])
                avg_mae = np.mean([m[1] for m in all_metrics])
                avg_r2 = np.mean([m[2] for m in all_metrics])
                print(f"\nGrid Search Results | lr={lr}, batch_size={batch_size}, dropout={dropout} | Avg RMSE: {avg_rmse:.4f} MAE: {avg_mae:.4f} R2: {avg_r2:.4f}")

                # Ensemble prediction on validation sets
                # (For demonstration, use last fold's val_loader)
                ensemble_preds = ensemble_predict(fold_models[:args.ensemble_top_n], val_loader, args.device, use_fp16=args.use_fp16)
                y_true = torch.cat([y for _, _, y in val_loader])
                ensemble_rmse = np.sqrt(np.mean((y_true.numpy() - ensemble_preds.numpy()) ** 2))
                ensemble_r2 = 1 - np.sum((y_true.numpy() - ensemble_preds.numpy()) ** 2) / np.sum((y_true.numpy() - np.mean(y_true.numpy())) ** 2)
                print(f"Ensemble (top {args.ensemble_top_n}) RMSE: {ensemble_rmse:.4f} R2: {ensemble_r2:.4f}")

                if avg_r2 > best_overall_r2:
                    best_overall_r2 = avg_r2
                    best_hparams = (lr, batch_size, dropout)
                    best_models = fold_models[:args.ensemble_top_n]

    print(f"\nBest Hyperparameters: lr={best_hparams[0]}, batch_size={best_hparams[1]}, dropout={best_hparams[2]}")
    print(f"Best Avg R2: {best_overall_r2:.4f}")

    # Save best ensemble models
    for i, model in enumerate(best_models):
        torch.save(model.state_dict(), f"{args.save_path}_ensemble_{i+1}.pt")

if __name__ == "__main__":
    main()