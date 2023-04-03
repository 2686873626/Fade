"""
V2G training scripts
written by Telephone, 2022/4/27
"""
import torch
import pytorch_lightning as pl
import argparse
import os

from omegaconf import OmegaConf as OC
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from training.model import Model
from training.data import FFData


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=int,
                        default=[0])
    parser.add_argument('--config', type=str,
                        default='configs/config_video.yaml')
    parser.add_argument('--seed', type=int,
                        default=1024)
    parser.add_argument('--test', action='store_true',
                        default=False)
    parser.add_argument('--amp', action='store_true',
                        default=False)
    return parser.parse_args()


def get_callbacks():
    early_stop = EarlyStopping(monitor='val/auc', mode='max', patience=10)
    epoch_checkpoint = ModelCheckpoint(save_on_train_epoch_end=True)
    val_acc_checkpoint = ModelCheckpoint(
        monitor='val/acc',
        mode='max',
        filename='val_acc-epoch_{epoch:03d}-loss_{val/loss:.3f}-acc_{val/acc:.5f}-auc_{val/auc:.5f}',
        auto_insert_metric_name=False,
        # save_top_k=3,
    )
    val_auc_checkpoint = ModelCheckpoint(
        monitor='val/auc',
        mode='max',
        filename='val_auc-epoch_{epoch:03d}-loss_{val/loss:.3f}-acc_{val/acc:.5f}-auc_{val/auc:.5f}',
        auto_insert_metric_name=False,
        # save_top_k=3,
    )
    lr_monitor = LearningRateMonitor()
    return [epoch_checkpoint, val_acc_checkpoint, val_auc_checkpoint, early_stop, lr_monitor]


def run(args, opt):
    pl.seed_everything(args.seed, workers=True)
    model = Model(opt.data, opt.training)
    # model.load_state_dict(torch.load('logs/PartialGCN/LQ/lightning_logs/version_5290/checkpoints/val_acc-epoch_009-loss_0.377-acc_0.93119-auc_0.94736.ckpt', map_location='cpu')['state_dict'], False)
    # torch.save(model.backbone.backbone.state_dict(), 'effnet_b4_320_LQ.ckpt')
    # return 0
    callbacks = get_callbacks()
    ffdata = FFData(opt.data)
    trainer = pl.Trainer(
        logger=not args.test,
        default_root_dir=os.path.join(opt.training.save_dir, opt.training.model.model_name, opt.data.img_quality),
        # devices=len(args.gpus),
        gpus=args.gpus,
        strategy='ddp_find_unused_parameters_false',
        sync_batchnorm=True,
        precision=opt.training.precision,
        max_epochs=opt.training.num_epochs,
        log_every_n_steps=opt.training.log_interval,
        accumulate_grad_batches=opt.training.grad_accumulate,
        # benchmark=True,
        check_val_every_n_epoch=opt.training.val_interval,
        gradient_clip_algorithm='value',
        gradient_clip_val=5.0,
        detect_anomaly=True,
        callbacks=callbacks
    )

    if args.test:
        if os.path.exists(opt.training.resume):
            trainer.test(model, datamodule=ffdata, ckpt_path=opt.training.resume)
        else:
            trainer.test(model, datamodule=ffdata)

    else:
        if os.path.exists(opt.training.resume):
            trainer.fit(model, datamodule=ffdata, ckpt_path=opt.training.resume)
        else:
            trainer.fit(model, datamodule=ffdata)


if __name__ == '__main__':
    args = get_config()
    opt = OC.load(args.config)
    opt.training.num_samples = opt.data.num_samples
    opt.training.img_size = opt.data.img_size
    # if amp is used
    if args.amp:
        opt.training.precision = 16
        opt.data.train_batch_size *= 2
        opt.data.val_batch_size *= 2
        opt.training.optim.eps = 1.0e-3

    OC.set_readonly(opt, True)
    run(args, opt)