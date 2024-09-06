import os
import argparse
import pickle

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import UnsupervisedPretrain
from utils import CamCANUnsupervisedLoader, collate_fn_camcan_pretrain

     
class LitModel_supervised_pretrain(pl.LightningModule):
    def __init__(self, args, save_path):
        super().__init__()
        self.args = args
        self.save_path = save_path
        self.T = 0.2
        self.model = UnsupervisedPretrain(emb_size=256, heads=8, depth=4, n_channels=306, n_fft=125) # 306 channels for CamCAN; 125 n_fft due to 0.5s 250Hz (125 sample sample rate)
        
    def training_step(self, batch, batch_idx):

        # store the checkpoint every 5000 steps
        if self.global_step % 2000 == 0:
            self.trainer.save_checkpoint(
                filepath=f"{self.save_path}/epoch={self.current_epoch}_step={self.global_step}.ckpt"
            )

        samples = batch.float()
        contrastive_loss = 0

        masked_emb, samples_emb = self.model(samples, 0)
        samples_emb = F.normalize(samples_emb, dim=1, p=2)
        masked_emb = F.normalize(masked_emb, dim=1, p=2)
        N = samples.shape[0]

        logits = torch.mm(samples_emb, masked_emb.t()) / self.T
        labels = torch.arange(N).to(logits.device)
        contrastive_loss += F.cross_entropy(logits, labels, reduction="mean")

        self.log("train_loss", contrastive_loss)
        return contrastive_loss


    def configure_optimizers(self):
        # set optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )

        # set learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.3
        )

        return [optimizer], [scheduler]

    
def prepare_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # define the CamCAN dataloader
    loader = CamCANUnsupervisedLoader()
    train_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        drop_last=True,
        collate_fn=collate_fn_camcan_pretrain,
    )
    
    return train_loader
 
 
def pretrain(args):
    
    # get data loaders
    train_loader = prepare_dataloader(args)
    
    save_prefix = "/data/engs-pnpl/lina4368/experiments/BIOT/weights"
    # define the trainer
    N_version = (
        len(os.listdir(os.path.join(save_prefix))) + 1
    )
    # define the model
    save_path = f"{save_prefix}/{N_version}-unsupervised/checkpoints"
    
    model = LitModel_supervised_pretrain(args, save_path)
    
    # logger = TensorBoardLogger(
    #     save_dir="/data/engs-pnpl/lina4368/experiments/BIOT/logs",
    #     version=f"{N_version}/checkpoints",
    #     name="log-pretrain",
    # )
    logger = WandbLogger(
        project="BIOT",
        name=f"unsupervised-pretrain-{N_version}",
        save_dir="/data/engs-pnpl/lina4368/experiments/BIOT/logs",
    )
    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        # auto_select_gpus=True,
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=args.epochs,
    )

    # train the model
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--num_workers", type=int, default=32, help="number of workers")
    args = parser.parse_args()
    print (args)

    pretrain(args)
    
    
    