
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import KFold

from fair_bo_dissertation.datasets import AdultDataset
from simple_classifier import SimpleClassifier

from pathlib import Path
from typing import Literal


class AutomaticTrainer:
    """When called, this class trains a model with a certain dataset and returns a list of metrics"""

    DATA_DIR = Path('data')

    def __init__(self,
                 dataset: Literal['adult_census', 'german_credit'] = 'adult_census',
                 model: Literal['simple'] = 'simple',
                 metrics=['acc', 'diff_tpr'],
                 n_splits=5):

        self.n_splits = n_splits

        # Load dataset
        if dataset == 'adult_census':
            self.dataset = AdultDataset(self.DATA_DIR / 'adult-census-income' / 'adult-processed.csv')
        else:
            raise ValueError(f'The dataset {dataset} is not implemented')

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)
        self.kf_splits = kf.split(self.dataset)

        # Load model
        if model == 'simple':
            self.model_func = SimpleClassifier
        else:
            raise ValueError(f'The model {model} is not implemented')

        # Metrics
        if metrics != ['acc', 'diff_tpr']:
            raise ValueError(f'At the moment, the only allowed metrics are [acc, diff_tpr]. Please implement your '
                             f'metric to be able to use it.')



    def __call__(self, x):

        total_confmat = torch.zeros((2, 2))
        protected_confmats = {i: torch.zeros((2, 2)) for i in range(2)}

        for idx_train, idx_val in self.kf_splits:

            # Get model

            model = self.model_func(len(self.dataset[0][0]))

            # Prepare data

            # train_len = int(0.8 * len(self.dataset))
            # train_dataset, val_dataset = random_split(self.dataset, [train_len, len(self.dataset) - train_len])

            train_dataset = self.dataset.create_subset(idx_train)
            val_dataset = self.dataset.create_subset(idx_val)

            train_loader = DataLoader(train_dataset,
                                      batch_size=32,
                                      shuffle=False,
                                      num_workers=4)
            val_loader = DataLoader(val_dataset,
                                    batch_size=32,
                                    shuffle=False,
                                    num_workers=4)

            # Train

            trainer = pl.Trainer(max_epochs=2)

            trainer.fit(model,
                        train_loader)

            # Validate
            trainer.validate(model,
                             val_loader)

            # Accumulate metrics
            total_confmat += model.validation_confmat
            for k, v in model.validation_protected_confmats.items():
                protected_confmats[k] += v


        # Obtain metrics
        acc = torch.sum(torch.diag(total_confmat)) / torch.sum(total_confmat)
        tprs = torch.stack([(v[1, 1] / torch.sum(v[1, :])) for v in protected_confmats.values()])
        tpr_diff = torch.max(tprs) - torch.min(tprs)

        return torch.stack((acc, tpr_diff))


if __name__ == '__main__':
    at = AutomaticTrainer(dataset='adult_census')
    out = at(None)
    print(out)
