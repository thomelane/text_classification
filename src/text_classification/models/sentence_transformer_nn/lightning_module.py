from typing import Any, Dict, List

import torch

import pytorch_lightning as pl
import torchmetrics


class ClassificationHead(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout: float,
        learning_rate: float,
        momentum: float,
        weight_decay: float
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.init_model(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout
        )
        self.init_metrics(num_classes)
        self.loss = torch.nn.CrossEntropyLoss()
        self.dropout = dropout
        # training hyper-parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

    def init_metrics(self, num_classes: int):
        self.train_loss = torchmetrics.MeanMetric()
        self.valid_loss = torchmetrics.MeanMetric()
        self.best_valid_f1_score = 0.  # used for TensorBoardLogger's hp_metric
        self.train_f1_score = torchmetrics.F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="macro"
        )
        self.valid_f1_score = torchmetrics.F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="macro"
        )
    
    def init_model(
            self,
            input_dim: int,
            hidden_dims: List[int],
            num_classes: int,
            dropout: float
        ):
        model = torch.nn.Sequential()
        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(len(dims) - 1):
            model.add_module(
                f"relu_{i}",
                torch.nn.ReLU()
            )
            model.add_module(
                f"dropout_{i}",
                torch.nn.Dropout(p=dropout)
            )
            model.add_module(
                f"linear_{i}",
                torch.nn.Linear(dims[i], dims[i+1])
            )
        return model

    def training_step(self, batch, _):
        embeddings, y = batch
        preds = self(embeddings)
        loss = self.loss(preds, y)
        self.train_loss.update(loss)
        self.log('train_loss', self.train_loss, on_step=True, on_epoch=True)
        self.train_f1_score.update(preds, y)
        self.log('train_f1_score', self.train_f1_score, on_step=True, on_epoch=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)
        return loss

    def validation_step(self, batch, _):
        embeddings, y = batch
        preds = self(embeddings)
        loss = self.loss(preds, y)
        self.valid_loss.update(loss)
        self.log('valid_loss', self.valid_loss, on_step=True, on_epoch=True)
        self.valid_f1_score.update(preds, y)
        self.log('valid_f1_score', self.valid_f1_score, on_step=True, on_epoch=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=0.2,
            div_factor=10,
            final_div_factor=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step"
            }
        }

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.model(embeddings)
    
    def on_validation_epoch_end(self) -> None:
        valid_f1_score = self.valid_f1_score.compute()
        self.best_valid_f1_score = max(self.best_valid_f1_score, valid_f1_score)
        self.log('hp_metric', self.best_valid_f1_score)
