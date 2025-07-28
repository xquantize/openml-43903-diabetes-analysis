import os
import sys

import yaml
import pandas as pd

import torch
import torch.nn as nn

import datetime
import matplotlib.pyplot as plt

import importlib
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data.dataloader import get_dataloaders


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

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return roc_auc_score(targets, preds)


def main():
    cfg = load_config()
    m_cfg = cfg['model']
    module_name = m_cfg.get('module', 'transformer')
    class_name = m_cfg.get('class', 'TabularTransformer')

    model_mod = importlib.import_module(f"research.models.{module_name}")
    ModelClass = getattr(model_mod, class_name)
    model_id = f"{module_name}.{class_name}"

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{model_id}_d{m_cfg['d_model']}_h{m_cfg['n_heads']}_l{m_cfg['n_layers']}_{ts}"
    run_dir = os.path.join("experiments", run_name)
    os.makedirs(run_dir, exist_ok=True)
    curves_dir = os.path.join(run_dir, "curves"); os.makedirs(curves_dir, exist_ok=True)
    tb_dir = os.path.join(run_dir, "tensorboard")

    writer = SummaryWriter(log_dir=tb_dir)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    df = pd.read_csv(cfg['data']['path'])
    cat_cols = cfg['data'].get('categorical', [
        'race', 'gender', 'age', 'admission_source_id',
        'medical_specialty', 'primary_diagnosis',
        'max_glu_serum', 'A1Cresult', 'insulin', 'change', 'diabetesMed'
    ])
    for col in cat_cols:
        df[col] = df[col].fillna('Missing').astype('category')
    cardinalities = {c: len(df[c].cat.categories) for c in cat_cols}

    num_cols = cfg['data'].get('numeric', [
        'time_in_hospital', 'num_lab_procedures',
        'num_procedures', 'num_medications', 'number_diagnoses'
    ])

    bool_cols = cfg['data'].get('boolean', [
        'medicare', 'medicaid',
        'had_emergency', 'had_inpatient_days', 'had_outpatient_days'
    ])

    n_numeric = len(num_cols) + len(bool_cols)

    train_loader, val_loader = get_dataloaders(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelClass(cardinalities, n_numeric, cfg).to(device)

    lr = float(cfg['training']['lr'])
    pos_w = torch.tensor(float(cfg['training']['pos_weight']), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_aucs = [], []
    best_auc, stagnate = 0.0, 0
    patience = cfg['training'].get('patience', 5)

    for epoch in range(1, cfg['training']['epochs']+1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        va_auc  = eval_model(model, val_loader, device)
        train_losses.append(tr_loss)
        val_aucs.append(va_auc)

        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("AUC/val", va_auc, epoch)
        print(f"Epoch {epoch}/{cfg['training']['epochs']}: loss={tr_loss:.4f}, val_auc={va_auc:.4f}")

        if va_auc > best_auc:
            best_auc = va_auc
            stagnate = 0
            ckpt_path = os.path.join(run_dir, "checkpoint.pth")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'config': cfg,
                'val_auc': va_auc
            }, ckpt_path)
            print(f"  → New best AUC; checkpoint saved to {ckpt_path}")
        else:
            stagnate += 1
            if stagnate >= patience:
                print(f"No improvement for {patience} epochs; stopping early.")
                break

    writer.close()

    metrics = pd.DataFrame({
        'epoch': range(1, len(train_losses)+1),
        'train_loss': train_losses,
        'val_auc': val_aucs
    })
    metrics.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    plt.figure(); plt.plot(metrics.epoch, metrics.train_loss); plt.title("Train Loss")
    plt.savefig(os.path.join(curves_dir, "loss.png")); plt.close()
    plt.figure(); plt.plot(metrics.epoch, metrics.val_auc); plt.title("Val AUC")
    plt.savefig(os.path.join(curves_dir, "val_auc.png")); plt.close()

    print(f"Training complete! All artifacts saved under {run_dir}")


if __name__ == "__main__":
    main()
