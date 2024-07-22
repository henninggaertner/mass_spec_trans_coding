import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
from torchmetrics import F1Score, Accuracy, Recall, Precision, AUROC

HUB_MODELS = {
    #'mobilenet_v2': ("pytorch/vision:v0.10.0", "mobilenet_v2"),
    #'resnet18': ("pytorch/vision:v0.10.0", "resnet18"),
    'resnet101': ("pytorch/vision:v0.10.0", "resnet101"),
    #'resnet152': ("pytorch/vision:v0.10.0", "resnet152")

}
class HubModel(pl.LightningModule):
    def __init__(self, repo, hub_model_name, learning_rate=1e-3, freeze_base_model=True):
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
            nn.Linear(128, 1)
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(task='binary')
        self.f1 = F1Score(task='binary')
        self.recall = Recall(task='binary')
        self.auroc = AUROC(task='binary')


    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        logits = self(x).squeeze()
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().squeeze()
        logits = self(x).squeeze()
        loss = self.criterion(logits, y)
        preds = torch.sigmoid(logits)
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
