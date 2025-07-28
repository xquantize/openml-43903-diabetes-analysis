import os
import sys

import yaml
import pandas as pd

import torch
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from data.dataloader import MedicalDataset
from research.models.transformer import TabularTransformer


def load_config(path="research/config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt['config']

    cat_cols = [
        'race','gender','age','admission_source_id',
        'medical_specialty','primary_diagnosis',
        'max_glu_serum','A1Cresult','insulin','change','diabetesMed'
    ]
    num_cols = [
        'time_in_hospital','num_lab_procedures',
        'num_procedures','num_medications','number_diagnoses'
    ]
    bool_cols = [
        'medicare','medicaid',
        'had_emergency','had_inpatient_days','had_outpatient_days'
    ]

    df = pd.read_csv(cfg['data']['path'])

    for col in cat_cols:
        df[col] = df[col].fillna('Missing').astype('category')
    cardinalities = {c: len(df[c].cat.categories) for c in cat_cols}
    n_numeric      = len(num_cols) + len(bool_cols)

    model = TabularTransformer(cardinalities, n_numeric, cfg)
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    return model, cfg


def main(ckpt_path, output_csv="predictions.csv", threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = load_checkpoint(ckpt_path, device)

    df = pd.read_csv(cfg['data']['path'])

    cat_cols = [
        'race','gender','age','admission_source_id',
        'medical_specialty','primary_diagnosis',
        'max_glu_serum','A1Cresult','insulin','change','diabetesMed'
    ]
    num_cols  = [
        'time_in_hospital','num_lab_procedures',
        'num_procedures','num_medications','number_diagnoses'
    ]
    bool_cols = [
        'medicare','medicaid',
        'had_emergency','had_inpatient_days','had_outpatient_days'
    ]

    scaler = StandardScaler()

    dataset = MedicalDataset(df, cat_cols, num_cols, bool_cols, scaler)
    loader  = DataLoader(dataset, batch_size=cfg['data']['batch_size'], shuffle=False)

    all_probs = []
    with torch.no_grad():
        for x_cat, x_num, _ in loader:
            x_cat, x_num = x_cat.to(device), x_num.to(device)
            logits = model(x_cat, x_num)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)

    df['readmit_prob'] = all_probs
    df['readmit_pred'] = (df['readmit_prob'] >= threshold).astype(int)

    print("\nPrediction Summary:")
    print(df['readmit_prob'].describe(), "\n")

    if 'readmit_30_days' in df.columns:
        auc = roc_auc_score(df['readmit_30_days'], df['readmit_prob'])
        acc = accuracy_score(df['readmit_30_days'], df['readmit_pred'])
        print(f"AUC: {auc:.4f}  Accuracy (@{threshold}): {acc:.4f}\n")
        print("Classification Report:")
        print(classification_report(df['readmit_30_days'], df['readmit_pred']))

    df.to_csv(output_csv, index=False)
    print(f"\nSaved predictions to {output_csv}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="Path to model .pth file")
    p.add_argument("--output", default="predictions.csv",
                   help="Path to save CSV with probabilities and preds")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Classification threshold")
    args = p.parse_args()
    main(args.checkpoint, args.output, args.threshold)
