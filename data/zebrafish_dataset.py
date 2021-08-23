## Skeleton for Lightning DataModule to load the dataset.

class ZebrafishDataModule(LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage = None):
        # Make assignments here (val/train/test split).
        # Called on every process in Distributed Data Processing
        pass

    def train_dataloader(self):
        zebrafish_train = DataLoader(self.zebrafish_train,
                batch_size=self.batch_size)
        return zebrafish_train

    def val_dataloader(self):
        zebrafish_val = DataLoader(self.zebrafish_val,
                batch_size=self.batch_size)
        return zebrafish_val

    def test_dataloader(self):
        zebrafish_test = DataLoader(self.zebrafish_test, 
                batch_size=self.batch_size)
        return zebrafish_test

