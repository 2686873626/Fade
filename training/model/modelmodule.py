import json

import torch
import pytorch_lightning as pl

from torch.nn import functional as F
from torchmetrics.classification import Accuracy, AUROC
from pytorch_lightning import loggers as pl_loggers
from torchvision.utils import make_grid
import sys

# from .v2g import V2G
# from .v2g_conv import V2G
from .features import Xception_Fea
from pretrainedmodels import xception
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
        # self.train_auc = AUROC(num_classes=2, compute_on_step=False, dist_sync_on_step=True)
        self.eval_acc = Accuracy(dist_sync_on_step=True)
        self.eval_auc = AUROC(num_classes=2, compute_on_step=False, dist_sync_on_step=True)

    def forward(self, x, rois):
        return self.backbone(x, rois)

    def training_step(self, batch, batch_idx):
        inputs, targets, rois = self.batch_preprocess(batch)
        preds = self(inputs, rois)
        loss = self.criterion(preds, targets)
        self.train_acc(preds, targets)
        # self.train_auc(preds, targets)
        self.log_dict(
            {'train/acc': self.train_acc},
            on_step=False, on_epoch=True, sync_dist=True,
            rank_zero_only=True
        )
        # self.log_dict(
        #     {'train/acc':self.train_acc, 'train/auc':self.train_auc},
        #     on_step=False, on_epoch=True, sync_dist=True,
        #     rank_zero_only=True
        # )
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
            {'val/loss': loss, 'val/acc': self.eval_acc, 'val/auc': self.eval_auc},
            on_step=False, on_epoch=True, sync_dist=True,
            rank_zero_only=True
        )

    def test_step(self, batch, batch_idx):
        inputs, targets, rois = self.batch_preprocess(batch)
        preds = self(inputs, rois)
        self.eval_acc(preds, targets)
        self.eval_auc(preds, targets)
        self.log_dict(
            {'test/acc': self.eval_acc, 'test/auc': self.eval_auc},
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

        params = [{'params': params1}, {'params': params2, 'lr': optim_config.lr * 0.2},
                  {'params': params3, 'lr': 2e-5}]

        if 'adam' == optim_config.optimizer:
            optimizer = torch.optim.AdamW(params, lr=optim_config.lr, eps=optim_config.eps,
                                          weight_decay=optim_config.weight_decay)
        elif 'sgd' == optim_config.optimizer:
            optimizer = torch.optim.SGD(params, lr=optim_config.lr, momentum=optim_config.momentum,
                                        weight_decay=optim_config.weight_decay)
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
            lr_scheduler_config = {'scheduler': scheduler, 'monitor': 'val/acc',
                                   'frequency': self.hparams.training.val_interval}
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
                T_max=self.hparams.training.num_epochs * 700,
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
        # rois.unsqueeze_(1)
        # rois = rois.repeat(1, T, 1, 1) # same roi for each T frame in one sequence
        rois = rois.view(B * T * count_areas, 4)
        rois = F.pad(rois, (1, 0))  # left padding for roi index
        for i in range(B * T * count_areas):
            rois[i][0] = i // count_areas

        return inputs, labels, rois

    def log_tb_images(self, viz_images, rois) -> None:
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError('TensorBoard Logger not found')

        region_name = ['l_eye_bbox', 'r_eye_bbox', 'l_cheek_bbox', 'r_cheek_bbox', 'nose_bbox', 'mouth_bbox',
                       'whole_face']

        # Log the images (Give them different names)
        for sequence_idx, sequence in enumerate(viz_images):
            roi = rois[sequence_idx]
            for frame_idx, frame in enumerate(sequence):
                roi_frame = roi[frame_idx]
                for region_idx, region in enumerate(region_name):
                    roi_bbox = roi_frame[region_idx]
                    image = frame[:, int(roi_bbox[1] * frame.shape[1]):int(roi_bbox[3] * frame.shape[1]),
                            int(roi_bbox[0] * frame.shape[2]):int(roi_bbox[2] * frame.shape[2])]
                    tb_logger.add_image(f"Image/Sequence:{sequence_idx}_Frame:{frame_idx}_Region{region}", image, 0)

        sys.exit(0)

    def log_inter_images(self, viz_images) -> None:
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError('TensorBoard Logger not found')

        # Log the images (Give them different names)
        for roi_idx, roi in enumerate(viz_images):
            grid = make_grid(roi.unsqueeze(1))
            tb_logger.add_image(f"Image/Region{roi_idx}", grid, 0)

        sys.exit(0)


class BackboneModel(pl.LightningModule):
    def __init__(self, data, training):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = xception(num_classes=2, pretrained=False)
        self.backbone.load_state_dict(torch.load('/irip/tanlingfeng_2020/v2g/init_weights/xception_HQ_299.ckpt'))

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_acc = Accuracy(dist_sync_on_step=True)
        # self.train_auc = AUROC(num_classes=2, compute_on_step=False, dist_sync_on_step=True)
        self.eval_acc = Accuracy(dist_sync_on_step=True)
        self.eval_auc = AUROC(num_classes=2, compute_on_step=False, dist_sync_on_step=True)
        self.test_record = {"Original": {}, "Deepfakes": {}, "Face2Face": {}, "FaceSwap": {}, "NeuralTextures": {}}
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        inputs, labels, *_ = batch
        B, T, C, H, W = inputs.shape
        inputs = inputs.view(B * T, C, H, W)
        labels = labels.unsqueeze_(1)
        labels = labels.repeat(1, T).view(B * T)
        preds = self(inputs)
        loss = self.criterion(preds, labels)
        self.train_acc(preds, labels)
        self.log_dict(
            {'train/acc': self.train_acc},
            on_step=False, on_epoch=True, sync_dist=True,
            rank_zero_only=True
        )
        self.log('train/loss', loss,
                 on_step=True, on_epoch=True, sync_dist=True,
                 rank_zero_only=True
                 )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, *_ = batch
        B, T, C, H, W = inputs.shape
        inputs = inputs.view(B * T, C, H, W)
        labels = labels.unsqueeze_(1)
        labels = labels.repeat(1, T).view(B * T)
        preds = self(inputs)
        loss = self.criterion(preds, labels)
        self.eval_acc(preds, labels)
        self.eval_auc(preds, labels)
        self.log_dict(
            {'val/loss': loss, 'val/acc': self.eval_acc, 'val/auc': self.eval_auc},
            on_step=False, on_epoch=True, sync_dist=True,
            rank_zero_only=True
        )

    def test_step(self, batch, batch_idx):
        inputs, labels, fakeType, video_folder, frame_index = batch
        B, T, C, H, W = inputs.shape
        inputs = inputs.view(B * T, C, H, W)
        labels = labels.unsqueeze_(1)
        labels = labels.repeat(1, T).view(B * T)
        preds = self(inputs)
        self.eval_acc(preds, labels)
        self.eval_auc(preds, labels)
        self.log_dict(
            {'test/acc': self.eval_acc, 'test/auc': self.eval_auc},
            on_step=False, on_epoch=True, sync_dist=True,
            rank_zero_only=True
        )
        preds_record = self.softmax(preds).cpu().numpy().tolist()
        frame_index = frame_index.cpu().numpy().tolist()
        for index, pred in enumerate(preds_record):
            if video_folder[index] not in self.test_record[fakeType[index]]:
                self.test_record[fakeType[index]][video_folder[index]] = {}
            self.test_record[fakeType[index]][video_folder[index]][frame_index[index]] = pred[0]

    def on_test_epoch_end(self) -> None:
        with open("test_record.json", "w") as f:
            json.dump(self.test_record, f)

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

        params = [{'params': params1}, {'params': params2, 'lr': optim_config.lr * 0.2},
                  {'params': params3, 'lr': 2e-5}]

        if 'adam' == optim_config.optimizer:
            optimizer = torch.optim.AdamW(params, lr=optim_config.lr, eps=optim_config.eps,
                                          weight_decay=optim_config.weight_decay)
        elif 'sgd' == optim_config.optimizer:
            optimizer = torch.optim.SGD(params, lr=optim_config.lr, momentum=optim_config.momentum,
                                        weight_decay=optim_config.weight_decay)
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
            lr_scheduler_config = {'scheduler': scheduler, 'monitor': 'val/acc',
                                   'frequency': self.hparams.training.val_interval}
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
                T_max=self.hparams.training.num_epochs * 700,
                eta_min=optim_config.min_lr,
            )
            lr_scheduler_config = {'scheduler': scheduler, 'interval': 'step'}
        else:
            print('Unknown lr scheduler')
            NotImplementedError

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
