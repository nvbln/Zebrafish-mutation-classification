from torch.utils.data import Dataset

import torch
import flammkuchen as fl
import json
import math
import scipy
import numpy as np

class FreeSwimmingDataset(Dataset):
    """Free Swimming dataset. Uses free swimming behaviour with a
    static background stimulus."""

    def __init__(self, 
                 fish_dictionaries, 
                 transform=None,
                 sampling_frequency=None
                ):
        """
        Args:
            fish_dictionaries (list): list containing dictionaries (one
                for every fish). Dictionary should contain a 'behaviour'
                key with the path to the behaviour .hdf5 file and a
                'metadata' key with the path to the metadata.json file.
            transform (callable; optional): Optional transform to be applied
                on a sample.
        """
        self.fish_dictionaries = fish_dictionaries
        self.transform = transform
        self.sampling_frequency = sampling_frequency
        
        # Load the metadata in advance, since this is quite lightweight.
        # Then build a dictionary with this data and the behaviour file
        # path. The behaviour itself will then only be loaded on the
        # __getitem__ call.
        for dictionary in self.fish_dictionaries:
            with open(dictionary['metadata']) as metadata_file:
                metadata_json = json.load(metadata_file)

                # Add relevant metadata entries to the dictionary.
                dictionary['genotype'] = \
                        metadata_json['general']['animal']['genotype']
                dictionary['fish_id'] = metadata_json['general']['fish_id']

    def __len__(self):
       return len(self.fish_dictionaries) 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dictionary = self.fish_dictionaries[idx]

        # Load the datafile.
        behaviour = fl.load(dictionary['behaviour'], '/data')

        # Add the data to the dictionary as a numpy matrix.
        dictionary['behavioural_data'] = behaviour.to_numpy()

        # Cut out irrelevant columns.
        # Limit sequence length to not run out of memory
        # (will be done more appropriately in the future).
        dictionary['behavioural_data'] = dictionary['behavioural_data'][:100000,:15]

        # Apply transforms if any.
        if self.transform:
            dictionary['behavioural_data'] = \
                    self.transform(dictionary['behavioural_data'])

        # Resample the data such that all the sequences have the
        # same sampling frequency.
        if self.sampling_frequency:
            # TODO: Throw out samples with a sampling frequency lower
            # than the required sampling frequency.

            # Calculate the number of samples if we were to have the
            # desired sampling frequency.
            duration = behaviour['t'].iloc[-1]
            num_samples = duration * self.sampling_frequency

            dictionary['behavioural_data'] = scipy.signal.resample(
                    dictionary['behavioural_data'], round(num_samples))

        # Create a vector with the correct classification.
        y_len = dictionary['behavioural_data'].shape[0]
        if dictionary['genotype'] == 'TL':
            #y = torch.ones((y_len, 1), dtype=torch.double)
            y = np.ones((1,))
        else:
            #y = torch.zeros((y_len, 1), dtype=torch.double)
            y = np.zeros((1,))

        return (dictionary['behavioural_data'], y)
