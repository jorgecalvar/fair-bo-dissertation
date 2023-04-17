from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import torch

from fair_bo_dissertation.datasets.base_dataset import BaseDataset


class GermanCreditDataset(BaseDataset):
    target_col = 'target'
    protected_cols = ['sex', 'marital-status', 'age', 'foreign-worker']

    def __init__(self,
                 file_path,
                 protected_col='sex',
                 device='cpu'):
        self.file_path = Path(file_path)
        self.X = None
        self.y = None
        self.X_protected = None
        self.protected_col = protected_col
        self._load()

        # To device
        self.X.to(device=device)
        self.y.to(device=device)
        self.X_protected(device=device)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.X_protected[idx, self.protected_cols.index(self.protected_col)]

    def __len__(self):
        return len(self.X)

    def _load(self):
        df = pd.read_csv(self.file_path)
        categorical_cols = df.select_dtypes(object).drop(columns=self.protected_cols + [self.target_col],
                                                         errors='ignore').columns
        numerical_cols = df.select_dtypes(exclude=object).drop(columns=self.protected_cols + [self.target_col],
                                                               errors='ignore').columns

        # Process numerical
        self.scaler = StandardScaler()
        df_num = pd.DataFrame(
            self.scaler.fit_transform(df[numerical_cols]),
            index=df[numerical_cols].index,
            columns=df[numerical_cols].columns
        )

        # Process categorical
        df_cat = pd.get_dummies(df[categorical_cols], drop_first=True)

        # Process target
        self.target_encoder = LabelEncoder()
        y = self.target_encoder.fit_transform(df[self.target_col])

        # X
        X = np.concatenate(
            (
                df_num.to_numpy(),
                df_cat.to_numpy()
            ),
            axis=1
        )
        self.X = torch.tensor(X).float()

        # y
        self.y = torch.tensor(y).float()

        # Protected cols
        self.X_protected = torch.zeros((self.X.size()[0], len(self.protected_cols))).int()
        self.protected_cols_encoders = {}
        for i, protected_col in enumerate(self.protected_cols):
            encoder = LabelEncoder()
            y_protected = encoder.fit_transform(df[protected_col])
            self.protected_cols_encoders[protected_col] = encoder
            self.X_protected[:, i] = torch.tensor(y_protected)


if __name__ == '__main__':
    file_path = Path('data/german-credit-data/german-credit-processed.csv')
    german_dataset = GermanCreditDataset(file_path)

    print(len(german_dataset))
    print(german_dataset[5])
