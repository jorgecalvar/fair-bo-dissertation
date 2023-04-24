from pathlib import Path

from fair_bo_dissertation.datasets.base_dataset import BaseDataset

from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch


class CrimeDataset(BaseDataset):
    target_col = 'high-crime'
    protected_cols = ['racepctblack',
                      'racePctWhite',
                      'racePctAsian',
                      'racePctHisp',
                      'agePct12t21',
                      'agePct12t29',
                      'agePct16t24',
                      'agePct65up',
                      'pctWWage',
                      'whitePerCap',
                      'blackPerCap',
                      'indianPerCap',
                      'AsianPerCap',
                      'HispPerCap',
                      'MalePctDivorce',
                      'MalePctNevMarr',
                      'FemalePctDiv',
                      'black']

    def __init__(self,
                 file_path,
                 protected_col='black',
                 device='cpu'):
        super().__init__()
        self.file_path = Path(file_path)
        self.protected_col = protected_col
        self.X = None
        self.y = None
        self.X_protected = None
        self._load()

        # To device
        self.X = self.X.to(device=device)
        self.y = self.y.to(device=device)
        self.X_protected = self.X_protected.to(device=device)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.X_protected[idx, self.protected_cols.index(self.protected_col)]

    def __len__(self):
        return len(self.X)

    def _load(self):
        df = pd.read_csv(self.file_path)

        # Process numerical
        df_num = df.loc[:, map(lambda x: x not in (self.protected_cols + [self.target_col]), df.columns)]
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(df_num)
        self.X = torch.tensor(X).float()

        # Process target
        y = df[self.target_col]
        self.y = torch.tensor(y).float()

        # Protected cols
        self.X_protected = torch.zeros((self.X.size()[0], len(self.protected_cols)))
        for i, protected_col in enumerate(self.protected_cols):
            self.X_protected[:, i] = torch.tensor(df[protected_col])