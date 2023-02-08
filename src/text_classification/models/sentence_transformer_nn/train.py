import os
from typing import List

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from text_classification.models.sentence_transformer_nn.lightning_module import ClassificationHead


def train(
    version: str,
    model_name: str,
    hidden_dims: List[int],
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    max_epochs: int,
    batch_size: int,
    overfit_batches: float,
    dropout: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
):
    """
    Sets up the training components and then trains the model. Creates:
        - Datasets
        - DataLoaders
        - Checkpoint callback
        - TensorBoard logger
        - Trainer
    """
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    valid_dataset = TensorDataset(
        torch.from_numpy(X_valid).float(),
        torch.from_numpy(y_valid).long()
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    model = ClassificationHead(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        num_classes=len(np.unique(y_train)),
        dropout=dropout,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    assert os.environ["MODELS_ROOT"] is not None
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{os.environ['MODELS_ROOT']}/{model_name}/{version}",
        filename="{epoch:02d}_{valid_f1_score_epoch:.2f}",
        monitor="valid_f1_score_epoch",
        mode="max"
    )
    logger = TensorBoardLogger(
        save_dir=os.environ["MODELS_ROOT"],
        name=model_name,
        version=version
    )
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=max_epochs,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
        overfit_batches=overfit_batches
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader
    )
    return model
