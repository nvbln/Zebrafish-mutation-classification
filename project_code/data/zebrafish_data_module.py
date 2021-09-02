import os
import glob
import ntpath

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from torchvision import transforms

from .resting_dataset import RestingDataset

class ZebrafishDataModule(pl.LightningDataModule):
    def __init__(self, 
                 batch_size=32,
                 data_dir='./data',
                 sampling_frequency=None
                ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.sampling_frequency = sampling_frequency

    def setup(self, stage = None):
        # Make assignments here (val/train/test split).
        # Called on every process in Distributed Data Processing

        # Retrieve all the directories containing fish.
        fish_directories = os.listdir(self.data_dir)

        fish_dictionaries = []

        for fish_directory in fish_directories:
            # Get the right behaviour and metadata files.
            behaviour_path = self.data_dir + '/' + fish_directory \
                             + '/*_behavior_log.hdf5'
            metadata_path = self.data_dir + '/' + fish_directory \
                            + '/*_metadata.json'

            behaviour_filenames = glob.glob(behaviour_path)

            if len(behaviour_filenames) > 1:
                # There are multiple files.
                # This can arise due to two scenarios:
                # 1. The experiment was interrupted and restarted.
                # 2. The fish ID was not changed and two different fish
                #    ended up in the same directory.
                # We'll assume scenario 1 here. Thus we select
                # the biggest file.
                sizes = []
                for filename in behaviour_filenames:
                    sizes.append(os.path.getsize(filename))

                # Use the biggest size index to retrieve the filename.
                behaviour_file = behaviour_filenames[sizes.index(max(sizes))]
            if len(behaviour_filenames) == 1:
                # There is only one file, so we use the first index.
                behaviour_file = behaviour_filenames[0]
            else:
                # The directory does not contain any (related) files.
                # Move on to the next case.
                continue

            ## Get the matching metadata and stimulus file.

            # First get only the filename (so not path).
            # We use ntpath for this to stay OS independent.
            head, tail = ntpath.split(behaviour_file)
            behaviour_only_filename = head if not tail else tail

            # Next extract the ID which should be before the first _.
            behaviour_ID = behaviour_only_filename.split('_')[0]

            # Then we can extract the metadata and stimulus file using the ID:
            metadata_file = self.data_dir + '/' + fish_directory \
                            + '/' + behaviour_ID \
                            + '_metadata.json'
            stimulus_file = self.data_dir + '/' + fish_directory \
                            + '/' + behaviour_ID \
                            + '_stimulus_log.hdf5'

            # Data selection will be done in the Dataset.
            # Therefore we won't load the actual data here,
            # but just provide the file names.
            fish_dictionary = {'behaviour': behaviour_file,
                               'metadata': metadata_file,
                               'stimulus': stimulus_file}
            fish_dictionaries.append(fish_dictionary)

        # Create datasets out of the data
        # This is the place where we decide in what form we will
        # use our data. I.e. we can specify a type of dataset that
        # uses the resting period, or a type of dataset that uses
        # the stimulus period.
        dataset = RestingDataset(fish_dictionaries, 
                                 sampling_frequency=self.sampling_frequency)

        # Split the data, random split should not be a problem
        # if we specify the seed, as then the test set will always
        # be the same, and thus we'll never touch it, even if we
        # do the random split again.
        num_samples = len(dataset)
        num_train = round(num_samples * 0.8) # 80% of the data.
        num_val = round(num_samples * 0.1) # 10% of the data.
        num_test = round(num_samples * 0.1) # 10% of the data.

        # We could be missing samples due to rounding. 
        # In that case we add it to the test set.
        num_test = num_test +  num_samples - (num_train + num_val + num_test)

        self.train, self.val, self.test = \
                torch.utils.data.random_split(dataset, \
                [num_train, num_val, num_test])

    def train_dataloader(self):
        zebrafish_train = DataLoader(self.train, batch_size=self.batch_size)
        return zebrafish_train

    def val_dataloader(self):
        zebrafish_val = DataLoader(self.val, batch_size=self.batch_size)
        return zebrafish_val

    def test_dataloader(self):
        zebrafish_test = DataLoader(self.test, batch_size=self.batch_size)
        return zebrafish_test
