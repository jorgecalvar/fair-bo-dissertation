import pytorch_lightning as pl
import torch
import torchmetrics.functional
from torch.utils.data import DataLoader, random_split
import tqdm
from sklearn.model_selection import KFold
from concurrent.futures import ThreadPoolExecutor

from fair_bo_dissertation.datasets import AdultDataset
from fair_bo_dissertation.datasets import GermanCreditDataset
from .simple_classifier import SimpleClassifier

import logging
from pathlib import Path
from typing import Literal


class AutomaticTrainer:
    """When called, this class trains a model with a certain dataset and returns a list of metrics"""

    DATA_DIR = Path('data')

    def __init__(self,
                 dataset: Literal['adult_census', 'german_credit'] = 'adult_census',
                 model: Literal['simple'] = 'simple',
                 metrics=['acc', 'diff_tpr'],
                 input_vars=['dropout', 'lr'],
                 n_splits=5,
                 epochs=2,
                 calculate_epoch_metrics=False,
                 device='cpu'):

        self.n_splits = n_splits
        self.epochs = epochs
        self.calculate_epoch_metrics = calculate_epoch_metrics
        self.device = device
        assert self.device in ('cuda', 'cpu')

        # Load dataset
        if dataset == 'adult_census':
            self.dataset = AdultDataset(self.DATA_DIR / 'adult-census-income' / 'adult-processed.csv',
                                        device=self.device)
        elif dataset == 'german_credit':
            self.dataset = GermanCreditDataset(self.DATA_DIR / 'german-credit-data' / 'german-credit-processed.csv',
                                               device=self.device)
        else:
            raise ValueError(f'The dataset {dataset} is not implemented')

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)
        kf_splits = kf.split(self.dataset)

        self.kf_datasets = []

        for idx_train, idx_val in kf_splits:
            train_dataset = self.dataset.create_subset(idx_train)
            val_dataset = self.dataset.create_subset(idx_val)

            # train_loader = DataLoader(train_dataset,
            #                           batch_size=32,
            #                           shuffle=False,
            #                           num_workers=1)
            # val_loader = DataLoader(val_dataset,
            #                         batch_size=32,
            #                         shuffle=False,
            #                         num_workers=1)

            self.kf_datasets.append((train_dataset, val_dataset))

        # Load model
        if model == 'simple':
            self.model_func = SimpleClassifier
        else:
            raise ValueError(f'The model {model} is not implemented')

        # Metrics
        self.metrics = metrics
        if self.metrics != ['acc', 'diff_tpr']:
            raise ValueError(f'At the moment, the only allowed metrics are [acc, diff_tpr]. Please implement your '
                             f'metric to be able to use it.')

        # Check input vars
        self.input_vars = input_vars
        assert all([k in self.model_func.get_kwargs() for k in input_vars])

    def __call__(self, x):
        """
        Creates the model and trains using k-fold cross validation
        :param x: a tensor specifying the input
        :return: a tensor with the metrics
        """

        kwargs_normalized = {k: x[i].item() for i, k in enumerate(self.input_vars)}
        kwargs_unnormalized = self.model_func.unnormalize_kwargs(**kwargs_normalized)

        # print(kwargs_unnormalized)

        total_confmat = torch.zeros((2, 2), device=self.device)
        protected_confmats = {i: torch.zeros((2, 2), device=self.device) for i in range(2)}


        def train_kf_split(train_dataset,
                           val_dataset,
                           total_confmat,
                           protected_confmats):

            # Get model

            model = self.model_func(len(self.dataset[0][0]), **kwargs_unnormalized).to(device=self.device)

            # Prepare data

            # train_len = int(0.8 * len(self.dataset))
            # train_dataset, val_dataset = random_split(self.dataset, [train_len, len(self.dataset) - train_len])

            # train_dataset = self.dataset.create_subset(idx_train)
            # val_dataset = self.dataset.create_subset(idx_val)
            #
            # train_loader = DataLoader(train_dataset,
            #                           batch_size=32,
            #                           shuffle=False,
            #                           num_workers=4)
            # val_loader = DataLoader(val_dataset,
            #                         batch_size=32,
            #                         shuffle=False,
            #                         num_workers=4)

            # Train

            # trainer = pl.Trainer(max_epochs=2)

            # trainer.fit(model,
            #             train_loader)

            optimizer = model.configure_optimizers()
            batch_size = 32

            if self.calculate_epoch_metrics:
                train_loss = []
                train_acc = []

            for _ in range(self.epochs):
                for batch_idx in (range((len(train_dataset) - 1) // batch_size + 1)):
                    batch = train_dataset[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                    # for batch_idx, batch in enumerate(train_loader):
                    optimizer.zero_grad()
                    loss = model.training_step(batch, batch_idx)
                    loss.backward()
                    optimizer.step()
                if self.calculate_epoch_metrics:
                    train_loss.append(loss.detach().item())
                    x, y, _ = train_dataset[:]
                    model.eval()
                    with torch.no_grad():
                        pred_y = model(x).squeeze(-1)
                    model.train()
                    train_acc.append(torchmetrics.functional.accuracy(pred_y, y, 'binary'))


            # Validate

            # trainer.validate(model,
            #                  val_loader)

            model.eval()
            with torch.no_grad():
                model.on_validation_start()
                for batch_idx in range((len(val_dataset) - 1) // batch_size + 1):
                    batch = val_dataset[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                    # for batch_idx, batch in enumerate(val_loader):
                    model.validation_step(batch, batch_idx)

            # Accumulate metrics
            total_confmat += model.validation_confmat
            for k, v in model.validation_protected_confmats.items():
                protected_confmats[k] += v

        # Execute simultaneously
        pool = ThreadPoolExecutor(max_workers=5)
        for train_dataset, val_dataset in self.kf_datasets:
            pool.submit(train_kf_split, train_dataset, val_dataset, total_confmat, protected_confmats)
        pool.shutdown(wait=True, cancel_futures=False)

        # Obtain metrics
        acc = torch.sum(torch.diag(total_confmat)) / torch.sum(total_confmat)
        tprs = torch.stack([(v[1, 1] / torch.sum(v[1, :])) for v in protected_confmats.values()])
        tpr_diff = torch.min(tprs) - torch.max(tprs) + 1


        return torch.stack((acc, tpr_diff))


if __name__ == '__main__':
    logger = logging.getLogger()
    print(logger)
    quit()
    at = AutomaticTrainer(dataset='adult_census')
    out = at(torch.tensor([0.3, 0.8]))
    print(out)
