from typing import Dict, Optional

import torch
import pytorch_lightning as pl
from yacs.config import CfgNode

from .ffhgenerator_data_set import FFHGeneratorDataSet


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

    def train_dataloader(self) -> Dict:
        """
        Setup training data loader.
        Returns:
            Dict: Dictionary containing image and mocap data dataloaders
        """
        train_dataloader = FFHGeneratorDataSet(self.cfg, eval=False)
        return train_dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Setup val data loader.
        Returns:
            torch.utils.data.DataLoader: Validation dataloader
        """
        val_dataloader = FFHGeneratorDataSet(self.cfg, eval=True)
        return val_dataloader
