from typing import Any, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torchvision import models
from torchmetrics import F1Score, Accuracy, Recall, Precision, AUROC

HUB_MODELS = {
    #'mobilenet_v2': ("pytorch/vision:v0.10.0", "mobilenet_v2", ),
    #'resnet18': ("pytorch/vision:v0.10.0", "resnet18"),
    'resnet101': ("pytorch/vision:v0.10.0", "resnet101"),
    #'resnet152': ("pytorch/vision:v0.10.0", "resnet152"),
    #'inception_v3': ("pytorch/vision:v0.10.0", "inception_v3"),
}


class ValidationCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics['val_loss']
        trainer.logger.log_metrics({'val_loss': val_loss}, step=trainer.global_step)

class MLPClassifier(pl.LightningModule):
    def __init__(self, input_size, num_classes=2, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()

        # Add metrics as in HubModel
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.f1 = F1Score(task='multiclass', num_classes=num_classes)
        self.recall = Recall(task='multiclass', num_classes=num_classes)
        self.auroc = AUROC(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.softmax(logits, dim=1)
        self.log('test_loss', loss)
        self.log('test_f1', self.f1(preds, y))
        self.log('test_acc', self.accuracy(preds, y))
        self.log('test_recall', self.recall(preds, y))
        self.log('test_auroc', self.auroc(preds, y))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

class HubModel(pl.LightningModule):
    def __init__(self, repo, hub_model_name, learning_rate=1e-3, num_classes=2, freeze_base_model=True):
        super().__init__()
        self.save_hyperparameters()

        # Load the model from PyTorch Hub
        self.base_model = torch.hub.load(repo, hub_model_name, pretrained=True)

        # Freeze the base model parameters
        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Get the number of features in the last layer
        if hasattr(self.base_model, 'fc'):
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif hasattr(self.base_model, 'classifier'):
            num_features = self.base_model.classifier[-1].in_features
            self.base_model.classifier = nn.Identity()

        # Add 3 classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.f1 = F1Score(task='multiclass', num_classes=num_classes)
        self.recall = Recall(task='multiclass', num_classes=num_classes)
        self.auroc = AUROC(task='multiclass', num_classes=num_classes)

        self.train_losses = []
        self.val_losses = []


    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.softmax(logits, dim=1)
        self.log('train_loss', loss)
        self.log('train_f1', self.f1(preds, y))
        self.log('train_acc', self.accuracy(preds, y))
        self.log('train_recall', self.recall(preds, y))
        self.log('train_auroc', self.auroc(preds, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.softmax(logits, dim=1)
        f1 = self.f1(preds, y)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds, y)
        auroc = self.auroc(preds, y)
        self.log('val_loss', loss)
        self.log('val_f1', f1)
        self.log('val_acc', acc)
        self.log('val_recall', recall)
        self.log('val_auroc', auroc)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.softmax(logits, dim=1)
        f1 = self.f1(preds, y)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds, y)
        auroc = self.auroc(preds, y)
        self.log('test_loss', loss)
        self.log('test_f1', f1)
        self.log('test_acc', acc)
        self.log('test_recall', recall)
        self.log('test_auroc', auroc)
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.train_losses.append(avg_loss.item())
        self.log('train_loss_epoch', avg_loss, prog_bar=True)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.val_losses.append(avg_loss.item())
        self.log('val_loss_epoch', avg_loss, prog_bar=True)

    def get_losses(self):
        return self.train_losses, self.val_losses[1:] # there is validation done before training, so first one's invalid

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)