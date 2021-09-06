import numpy as np
import torch
import torch.nn as nn
import math

import pytorch_lightning as pl
import matplotlib.pyplot as plt

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
        self.linear = nn.Linear(20, 1)
        self.bceloss = nn.BCELoss()

    def forward(self, x, hiddens=None):
        # Do the forward pass.

        # Stop-gap solution: set nan to zero.
        x[np.isnan(x)] = 0

        if hiddens == None:
            out, hiddens = self.lstm(x)
        else:
            out, hiddens = self.lstm(x, hiddens)

        out = torch.sigmoid(out)
        out = self.linear(out)
        out = torch.sigmoid(out)

        return out, hiddens

    def training_step(self, train_batch, batch_idx):
        # Calculates the loss for a batch.
        x, y = train_batch

        y_hat, _ = self.forward(x)

        loss = self.bceloss(y_hat.double(), y.double())

        # Log to TensorBoard.
        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        # The same as in training_step,
        # but now we only log the loss and don't return it:
        x, y = val_batch

        y_hat, _ = self.forward(x)

        loss = self.bceloss(y_hat, y)

        # Log to TensorBoard
        self.log('val_loss', loss)

        # Log the confidence values to TensorBoard.
        fig = plt.figure()
        detached_y_hat = y_hat.detach()
        plt.plot(np.squeeze(detached_y_hat))
        self.logger.experiment.add_figure('confidence_values', 
                                          fig, 
                                          global_step=self.global_step)

    def configure_optimizers(self):
        # Return an optimizer (i.e. Adam).
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

