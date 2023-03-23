
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from fair_bo_dissertation.datasets import AdultDataset
from simple_classifier import SimpleClassifier

from pathlib import Path


def create_and_train_model():


    dataset = AdultDataset(Path('data/adult-census-income/adult-processed.csv'))
    train_len = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_len, len(dataset) - train_len])

    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=False)

    model = SimpleClassifier(len(dataset[0][0]))

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model,
                train_loader,
                val_loader)


def prueba():
    print('hola')


if __name__ == '__main__':
    create_and_train_model()

