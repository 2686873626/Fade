import torch
import pytorch_lightning as pl

from torch.nn import functional as F
from torchmetrics.classification import Accuracy, AUROC

# from .v2g import V2G
# from .v2g_conv import V2G
from .v2g_pattern_mixer_conv import V2G
# from .tcn import V2G
# from .ablation import V2G


count_areas = 7

class Model(pl.LightningModule):
    def __init__(self, data, training):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = V2G()

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_acc = Accuracy(dist_sync_on_step=True)
        self.train_auc = AUROC(num_classes=2, compute_on_step=False, dist_sync_on_step=True)
        self.eval_acc = Accuracy(dist_sync_on_step=True)
        self.eval_auc = AUROC(num_classes=2, compute_on_step=False, dist_sync_on_step=True)

    def forward(self, x, rois):
        return self.backbone(x, rois)

    def training_step(self, batch, batch_idx):
        inputs, targets, rois = self.batch_preprocess(batch)
        preds = self(inputs, rois)
        loss = self.criterion(preds, targets)
        self.train_acc(preds, targets)
        self.train_auc(preds, targets)
        self.log_dict(
            {'train/acc':self.train_acc, 'train/auc':self.train_auc},
            on_step=False, on_epoch=True, sync_dist=True,
            rank_zero_only=True
        )
        self.log('train/loss', loss,
            on_step=True, on_epoch=True, sync_dist=True,
            rank_zero_only=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, rois = self.batch_preprocess(batch)
        preds = self(inputs, rois)
        loss = self.criterion(preds, targets)
        self.eval_acc(preds, targets)
        self.eval_auc(preds, targets)
        self.log_dict(
            {'val/loss':loss, 'val/acc':self.eval_acc, 'val/auc':self.eval_auc},
            on_step=False, on_epoch=True, sync_dist=True,
            rank_zero_only=True
        )

    def test_step(self, batch, batch_idx):
        inputs, targets, rois = self.batch_preprocess(batch)
        preds = self(inputs, rois)
        self.eval_acc(preds, targets)
        self.eval_auc(preds, targets)
        self.log_dict(
            {'test/acc':self.eval_acc, 'test/auc':self.eval_auc},
            on_step=False, on_epoch=True, sync_dist=True,
            rank_zero_only=True
        )

    def configure_optimizers(self):
        optim_config = self.hparams.training.optim
        params = self.parameters()
        params1, params2, params3 = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                pass
            if 'features' in name:
                params2.append(param)
            elif 'gcn' in name:
                params3.append(param)
                pass
            else:
                params1.append(param)
        
        params = [{'params':params1}, {'params':params2, 'lr':optim_config.lr*0.2}, {'params': params3, 'lr': 2e-5}]
                
        if 'adam' == optim_config.optimizer:
            optimizer = torch.optim.AdamW(params, lr=optim_config.lr, eps=optim_config.eps, weight_decay=optim_config.weight_decay)
        elif 'sgd' == optim_config.optimizer:
            optimizer = torch.optim.SGD(params, lr=optim_config.lr, momentum=optim_config.momentum, weight_decay=optim_config.weight_decay)
        else:
            print('Unknown optimizer')
            NotImplementedError
        
        if 'plateau' == optim_config.sche.sche_type:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=optim_config.sche.factor,
                patience=optim_config.sche.patience_epochs,
                min_lr=optim_config.min_lr
            )
            lr_scheduler_config = {'scheduler': scheduler, 'monitor': 'val/acc', 'frequency':self.hparams.training.val_interval}
        elif 'step' == optim_config.sche.sche_type:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=optim_config.sche.decay_epochs,
                gamma=optim_config.sche.gamma,
            )
            lr_scheduler_config = {'scheduler': scheduler, 'interval': 'epoch'}
        elif 'cosine' == optim_config.sche.sche_type:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.training.num_epochs*700,
                eta_min=optim_config.min_lr,
            )
            lr_scheduler_config = {'scheduler': scheduler, 'interval': 'step'}
        else:
            print('Unknown lr scheduler')
            NotImplementedError
        
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def batch_preprocess(self, batch):
        inputs, labels, rois = batch
        # inputs: batch, frame number of sequence, channel, height, width
        B, T, C, H, W = inputs.shape
        inputs = inputs.view(B * T, C, H, W)
        rois.unsqueeze_(1)
        rois = rois.repeat(1, T, 1, 1) # same roi for each T frame in one sequence
        rois = rois.view(B * T * count_areas, 4)
        rois = F.pad(rois, (1, 0)) # left padding for roi index
        for i in range(B * T * count_areas):
            rois[i][0] = i // count_areas

        return inputs, labels, rois
