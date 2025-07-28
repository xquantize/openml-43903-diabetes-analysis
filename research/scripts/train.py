import os
import sys

import yaml
import pandas as pd

import torch
import torch.nn as nn

import datetime
from sklearn.metrics import roc_auc_score


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data.dataloader import get_dataloaders
from research.models.transformer import TabularTransformer


def load_config(path="research/config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for x_cat, x_num, y in loader:
        x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x_cat, x_num)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_cat.size(0)

    return total_loss / len(loader.dataset)


def eval_model(model, loader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x_cat, x_num, y in loader:
            x_cat, x_num = x_cat.to(device), x_num.to(device)
            logits = model(x_cat, x_num)
            preds.append(torch.sigmoid(logits).cpu())
            targets.append(y)

    preds   = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    return roc_auc_score(targets, preds)


def main():
    cfg = load_config()

    df = pd.read_csv(cfg['data']['path'])
    cat_cols = cfg['data'].get('categorical', [
        'race','gender','age','admission_source_id',
        'medical_specialty','primary_diagnosis',
        'max_glu_serum','A1Cresult','insulin','change','diabetesMed'
    ])

    for col in cat_cols:
        df[col] = df[col].fillna('Missing').astype('category')
    cardinalities = {col: len(df[col].cat.categories) for col in cat_cols}

    num_cols  = cfg['data'].get('numeric', [
        'time_in_hospital','num_lab_procedures',
        'num_procedures','num_medications','number_diagnoses'
    ])
    bool_cols = cfg['data'].get('boolean', [
        'medicare','medicaid',
        'had_emergency','had_inpatient_days','had_outpatient_days'
    ])

    n_numeric = len(num_cols) + len(bool_cols)

    train_loader, val_loader = get_dataloaders(cfg)

    model = TabularTransformer(cardinalities, n_numeric, cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    lr = float(cfg['training']['lr'])
    pos_weight = torch.tensor(float(cfg['training']['pos_weight']), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_auc = 0.0
    patience, no_improve = cfg['training'].get('patience', 5), 0
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    for epoch in range(1, cfg['training']['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_auc    = eval_model(model, val_loader, device)
        print(f"Epoch {epoch}/{cfg['training']['epochs']} - Loss: {train_loss:.4f} - Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            no_improve = 0
            m = cfg['model']

            fname = (
                f"tabtrans_d{m['d_model']}_h{m['n_heads']}_l{m['n_layers']}"
                f"_ff{m['dim_ff']}_dr{m['dropout']}_{timestamp}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'config': cfg,
                'val_auc': val_auc
            }, fname)

            print(f" Saved new best model to {fname}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No improvement for {patience} epochs. Early stopping.")
                break

    print("Training complete.")


if __name__ == "__main__":
    main()
