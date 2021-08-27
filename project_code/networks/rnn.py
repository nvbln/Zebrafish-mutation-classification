import numpy as np
import torch
import torch.nn as nn

import pytorch_lightning as pl

# Define a Recurrent Neural Network
class MutationNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Define the structure of the network
        # Note: this is a very provisional model for testing purposes
        self.lstm = nn.LSTM(input_size = 11, 
                            hidden_size = 20,
                            num_layers = 1,
                            batch_first=True)
        self.bceloss = nn.BCELoss()

    def forward(self, x, hiddens=None):
        # Do the forward pass.

        # TODO: We limit the sequence to 500 as it will result in NaNs
        # in long(er) sequences. This bug should be fixed.
        x = x[:, :500]

        if hiddens == None:
            out, hiddens = self.lstm(x)
        else:
            out, hiddens = self.lstm(x, hiddens)

        return torch.sigmoid(out), hiddens

    def training_step(self, train_batch, batch_idx):
        # Calculates the loss for a batch.
        x = train_batch['behavioural_data']

        y_hat, _ = self.forward(x)

        if train_batch['treatment'] == 'None':
            y = torch.ones(y_hat.shape, dtype=torch.double)    
        else:
            y = torch.zeros(y_hat.shape, dtype=torch.double)    
        loss = self.bceloss(y_hat.double(), y.double())

        # Log to TensorBoard.
        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        # The same as in training_step,
        # but now we only log the loss and don't return it:
        x = val_batch['behavioural_data']
        y_hat, _ = self.forward(x)

        if val_batch['treatment'] == 'None':
            y = torch.ones(y_hat.shape, dtype=torch.double)    
        else:
            y = torch.zeros(y_hat.shape, dtype=torch.double)    

        loss = self.bceloss(y_hat, y)

        # Log to TensorBoard
        self.log('val_loss', loss)

    def configure_optimizers(self):
        # Return an optimizer (i.e. Adam).
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

