import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import src.losses as losses
import segmentation_models_pytorch as smp

class SMP(pl.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        self.save_hyperparameters(config)
        self.num_classes = 3
        self.loss = getattr(losses, self.hparams['loss'])
        self.model = getattr(smp, self.hparams['model'])(
            encoder_name=self.hparams['backbone'],
            encoder_weights=self.hparams['pretrained'],
            in_channels=3,
            classes=self.num_classes,
        )

    def forward(self, x):
        return self.model(x)

    def iou(self, pr, gt, th=0.5, eps=1e-3):
        pr = (pr > th).to(torch.float32)
        gt = gt.to(torch.float32)

        intersection = torch.sum(gt * pr, dim=(2,3))
        union = torch.sum(gt, dim=(2,3)) + torch.sum(pr, dim=(2,3)) - intersection + eps
        ious = (intersection + eps) / union
        return torch.mean(ious, dim=(1,0))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_hat = torch.sigmoid(y_hat)
        iou = self.iou(y_hat, y)
        dice = self.dice(y_hat, y)
        self.log('loss', loss)
        self.log('iou', iou, prog_bar=True)
        self.log('dice', dice)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_hat = torch.sigmoid(y_hat)
        iou = self.iou(y_hat, y)
        dice = self.dice(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_iou', iou, prog_bar=True)
        self.log('dice', dice)

    def test_step(self, batch, batch_idx): 
        x, y = batch
        y_hat = self(x)
        iou = self.iou(y_hat, y)
        self.log('test_iou', iou, prog_bar=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(self.parameters(), lr=self.hparams.lr)
        if 'scheduler' in self.hparams:
            schedulers = [
                getattr(torch.optim.lr_scheduler, scheduler)(optimizer, **params)
                for scheduler, params in self.hparams.scheduler.items()
            ]
            return [optimizer], schedulers
        return optimizer
