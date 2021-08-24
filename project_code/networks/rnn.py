## Generic skeleton for RNN.
import torch.nn as nn

# Define a Recurrent Neural Network
class MutationNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the structure of the network
        pass

    def forward(self, x):
        # Do the forward pass.
        pass

    def configure_optimizers(self):
        # Return an optimizer (i.e. Adam).
        pass

    def training_step(self, train_batch, batch_idx):
        # Load the data from the train batch.
        # Predict y_hat.
        # Calculate the loss between y_hat and y.
        # Log the loss with self.log('train_loss', loss).
        # Return the loss.
        pass

    def validation_step(self, val_batch, batch_idx):
        # The same as in training_step,
        # but now we only log the loss and don't return it:
        # self.log('val_loss', loss)
        pass
