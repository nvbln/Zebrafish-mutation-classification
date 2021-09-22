import numpy as np
import torch
import torch.nn as nn
import math

import pytorch_lightning as pl
import matplotlib.pyplot as plt

# Define a Recurrent Neural Network
class MutationNet(pl.LightningModule):
    def __init__(self, config = None):
        super().__init__() 

        # Save the hyperparameters to the model checkpoint.
        self.save_hyperparameters(config)

        # Define the structure of the network
        # Note: this is a very provisional model for testing purposes
        self.lstm = nn.LSTM(input_size = self.hparams.input_size, 
                            hidden_size = self.hparams.hidden_size,
                            num_layers = self.hparams.num_layers,
                            batch_first=True)
        self.linear = nn.Linear(self.hparams.hidden_size, 1)
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
        plt.plot(np.squeeze(detached_y_hat[0]))
        self.logger.experiment.add_figure('confidence_values', 
                                          fig, 
                                          global_step=self.global_step)

    def test_step(self, test_batch, batch_idx):
        # This function works just like validation_step.
        # Since we just want a log in Tensorboard, we'll call
        # validation_step with the same parameters. Note that by default
        # the logs will be appended to the earlier validation logs.
        # To prevent this, one can reinitialise the trainer.
        self.validation_step(test_batch, batch_idx)

    def configure_optimizers(self):
        # Return an optimizer (i.e. Adam).
        optimizer = torch.optim.Adam(self.parameters(), 
                    lr=self.hparams.learning_rate)

        # Implement a learning rate scheduler.
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99),
            'interval': 'step'
        }

        return [optimizer], [lr_scheduler]

