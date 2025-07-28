import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MedicalDataset(Dataset):
    def __init__(self, df, cat_cols, num_cols, bool_cols, scaler):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.bool_cols = bool_cols
        df = df.copy()

        for col in cat_cols:
            df[col] = df[col].fillna('Missing').astype('category')

        self.cat_vals = df[cat_cols].apply(lambda x: x.cat.codes).values
        num_bool_df = df[num_cols + bool_cols].astype(float)
        self.num_vals = scaler.fit_transform(num_bool_df)
        self.y = df['readmit_30_days'].values


    def __len__(self):
        return len(self.y)


    def __getitem__(self, idx):
        x_cat = torch.tensor(self.cat_vals[idx], dtype=torch.long)
        x_num = torch.tensor(self.num_vals[idx], dtype=torch.float)
        y     = torch.tensor(self.y[idx], dtype=torch.float)
        return x_cat, x_num, y


def get_dataloaders(cfg):
    df = pd.read_csv(cfg['data']['path'])

    cat_cols  = [
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

    train_idx, val_idx = train_test_split(
        list(range(len(dataset))),
        test_size=cfg['data']['test_size'],
        stratify=dataset.y,
        random_state=42
    )

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds   = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds,
                              batch_size=cfg['data']['batch_size'],
                              shuffle=True)
    val_loader   = DataLoader(val_ds,
                              batch_size=cfg['data']['batch_size'],
                              shuffle=False)

    return train_loader, val_loader
