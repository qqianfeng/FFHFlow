from typing import Dict, Optional

import torch
import pytorch_lightning as pl
from yacs.config import CfgNode

from .dataloader import FFHGeneratorDataset


class FFHDataModule(pl.LightningDataModule):

    def __init__(self, cfg: CfgNode) -> None:
        """
        Initialize LightningDataModule for ProHMR training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
            dataset_cfg (CfgNode): Dataset configuration file
        """
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load datasets necessary for training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
        """
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Setup training data loader.
        Returns:
            Dict: Dictionary containing image and mocap data dataloaders
        """
        dset_gen = FFHGeneratorDataset(self.cfg, eval=False)
        train_dataloader = torch.utils.data.DataLoader(dset_gen,
                                                        batch_size=self.cfg.TRAIN.BATCH_SIZE,
                                                        shuffle=True,
                                                        drop_last=True,
                                                        num_workers=self.cfg.GENERAL.NUM_WORKERS)

        return train_dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Setup val data loader.
        Returns:
            torch.utils.data.DataLoader: Validation dataloader
        """
        dset_gen = FFHGeneratorDataset(self.cfg, eval=True)
        val_dataloader = torch.utils.data.DataLoader(dset_gen,
                                                        batch_size=self.cfg.TRAIN.BATCH_SIZE,
                                                        shuffle=True,
                                                        drop_last=True,
                                                        num_workers=self.cfg.GENERAL.NUM_WORKERS)
        return val_dataloader

    def val_dataset(self):
        """Only for eval script to get access to Dataset class in order to obstain ground truth grasps

        Returns:
            _type_: _description_
        """
        return FFHGeneratorDataset(self.cfg, eval=True)
    
    def train_dataset(self):
        """Only for eval script to get access to Dataset class in order to obstain ground truth grasps

        Returns:
            _type_: _description_
        """
        return FFHGeneratorDataset(self.cfg, eval=False)