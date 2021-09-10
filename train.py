
from config import UNetConfig
from model.unet import NestedUNet, NestedUNetPlus, NestedUNetPlusAdd
import numpy as np
import os
import os.path as osp

from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything, LightningDataModule
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from utils.dataset import SemiconDataset
from utils.loss import DiceBCELoss

seed_everything(42, workers=True)
cfg = UNetConfig()

class LitNestedUnet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        cfg = UNetConfig()
        self.nnunet = eval(cfg.model)(cfg)
        self.criterion = DiceBCELoss()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        masks = self.nnunet(x)
        return masks

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y,_ = batch
        # x = x.view(x.size(0), -1)
        # for deeplysupervised model, the output is a list of 4 tensors
        list_of_masks = self.nnunet(x)

        loss_batch = 0
        bce_batch = 0
        dice_batch = 0
        # TODO save figures
        # Should we use binarized output for loss calculation?
        for masks in list_of_masks:
            bce, dice_loss = self.criterion(masks, y)
            bce_batch += bce
            dice_batch += dice_loss
            loss_batch += bce
            loss_batch += dice_loss
        # Logging to TensorBoard by default
        self.log("train_loss", loss_batch)
        self.log("train_bce_loss", bce_batch)
        self.log("train_dice_loss", dice_batch)
        return {'loss': loss_batch, 'bce': bce, 'dice': dice_loss}

    def validation_step(self, batch, batch_idx):
        x, y,_ = batch
        list_of_masks = self.nnunet(x)
        loss_batch = 0
        bce_batch = 0
        dice_batch = 0
        for masks in list_of_masks:
            bce, dice_loss = self.criterion(masks, y)
            bce_batch += bce
            dice_batch += dice_loss
            loss_batch += bce
            loss_batch += dice_loss
        # Logging to TensorBoard by default
        self.log("Validation_loss", loss_batch)
        self.log("Validation_bce_loss", bce_batch)
        self.log("Validation_dice_loss", dice_batch)
        return {'Validation_loss': loss_batch, 'Validation_bce': bce, 'Validation_dice': dice_loss}

    def test_step(self, batch, batch_idx):
        x, _, name = batch
        # x, segment_row, segment_column, name = batch
        # save original figure for reference
        # img_name = str(name[0]) + "_ori_row_" + str(int(segment_row[0])) + '_column_' + str(int(segment_column[0])) + ".png"
        # output_img_dir = '.'
        # x [1,1, 2048, 2048]
        # _x = x * 255
        # _x = _x.squeeze(0).squeeze(0)
        # save_image(_x, osp.join(output_img_dir, img_name))

        # continue predictions
        list_of_masks = self.nnunet(x)
        average = False
        if average:
            for out_idx, masks in enumerate(list_of_masks):
                probs = torch.sigmoid(masks)
                _masks = []
                probs = probs.squeeze(0)
                prob_ind = 0
                for prob in probs:
                    mask = prob.squeeze().cpu().numpy()
                    mask = mask > cfg.out_threshold
                    image_idx = Image.fromarray((mask * 255).astype(np.uint8))
                    img_name_idx = str(name[0]) + "_cls_" + str(prob_ind) + "_out_" + str(out_idx) + ".png"
                    image_idx.save(osp.join(cfg.save_dir, img_name_idx))
                    prob_ind += 1
        else:
            probs = torch.sigmoid(list_of_masks[-1])
            _masks = []
            probs = probs.squeeze(0).squeeze(0) # squeeze once for nnunet, twice for nnunet pro
            prob_ind = 0
            for prob in probs:
                mask = prob.squeeze().cpu().numpy()
                mask = mask > cfg.out_threshold
                image_idx = Image.fromarray((mask * 255).astype(np.uint8))
                img_name_idx = str(name[0]) + "_cls_" + str(prob_ind) + ".png"
                image_idx.save(osp.join(cfg.save_dir, img_name_idx))
                prob_ind += 1
            # # For new test data
            # masks = list_of_masks[-1]
            # # for masks in list_of_masks:
            # probs = torch.sigmoid(masks)
            # probs = probs.squeeze(0)
            # prob_ind = 0
            # for prob in probs:
            #     mask = prob.squeeze().cpu().numpy()
            #     mask = mask > cfg.out_threshold
            #     image_idx = Image.fromarray((mask * 255).astype(np.uint8))
            #     img_name_idx = str(name[0]) + "_cls_" + str(prob_ind) + '_row_' + str(int(segment_row[0])) + '_column_' + str(int(segment_column[0])) + ".png"
            #     image_idx.save(osp.join(cfg.save_dir, img_name_idx))
            #     prob_ind += 1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class DataModule(LightningDataModule):
    def __init__(self, batch_size=cfg.batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = SemiconDataset(mode='Train')
        self.validation_dataset = SemiconDataset(mode='Validation')
        self.test_dataset = SemiconDataset(mode='Test')

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)
        return train_loader

    def val_dataloader(self):
        validation_loader = DataLoader(self.validation_dataset)
        return validation_loader

    def test_dataloader(self):
        # For now , use validation as test
        # test_loader = DataLoader(self.test_dataset)
        test_loader = DataLoader(self.validation_dataset)
        return test_loader


# # Training
nnunet = LitNestedUnet()
dm = DataModule(batch_size=cfg.batch_size)
# trainer = Trainer(gpus=[3], log_every_n_steps=1, deterministic=True, max_epochs=cfg.epochs)
trainer = Trainer(gpus=[0,3], accelerator='ddp', log_every_n_steps=1, deterministic=True, 
    max_epochs=cfg.epochs, gradient_clip_val=0.5, stochastic_weight_avg=True, 
    precision=16, check_val_every_n_epoch=3)
trainer.fit(nnunet, datamodule=dm)
trainer.test()

# # Testing
# nnunet = LitNestedUnet.load_from_checkpoint('lightning_logs/version_52/checkpoints/epoch=399-step=399.ckpt')
# trainer = Trainer(gpus=[3])
# dm = DataModule(batch_size=cfg.batch_size)
# trainer.test(nnunet, datamodule=dm)