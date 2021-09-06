from torch.utils.data import Dataset

import torch
import flammkuchen as fl
import json
import math
import scipy

class RestingDataset(Dataset):
    """Resting dataset. Uses free swimming behaviour without any stimuli
    as behavioural data. The data comes from the experimental part after
    the calibration and before any stimuli is given."""

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
                dictionary['species'] = \
                        metadata_json['general']['animal']['species']
                dictionary['treatment'] = \
                        metadata_json['general']['animal']['treatment']
                dictionary['fish_id'] = metadata_json['general']['fish_id']

        # To prevent having to load every stimulus file, we'll assume that
        # the experimental setup is the same for every fish and therefore
        # extracting the times from the first stimulus file should suffice.
        # We're interested in the start of the general and gain_lag
        # stimuli (the latter of which signals the end of the first
        # general stimuli portion).
        stim_file = fl.load(self.fish_dictionaries[0]['stimulus'])

        # Before the onset of a stimulus, the array contains NaNs.
        # Therefore we can simply search for the first non-NaN value.
        general_base_vel = stim_file['data']['general_cl1D_base_vel']
        for i in range(len(general_base_vel)):
            if not math.isnan(general_base_vel[i]):
                self.start_t = stim_file['data']['t'][i]
                break

        gain_lag_base_vel = stim_file['data']['gain_lag_cl1D_base_vel']
        for i in range(len(gain_lag_base_vel)):
            if not math.isnan(gain_lag_base_vel[i]):
                self.end_t = stim_file['data']['t'][i]
                break

    def __len__(self):
       return len(self.fish_dictionaries) 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dictionary = self.fish_dictionaries[idx]

        # Load the datafile.
        behaviour = fl.load(dictionary['behaviour'], '/data')

        # Get the indices belonging to start_t and end_t.
        start_index_behaviour = next(index for index, value in \
                                     enumerate(behaviour['t']) \
                                     if value >= self.start_t)
        end_index_behaviour = next(index for index, value in \
                                   enumerate(behaviour['t']) \
                                   if value >= self.end_t)

        # Add the data to the dictionary (we leave t out).
        dictionary['behavioural_data'] = \
                behaviour[:-1][start_index_behaviour:end_index_behaviour]

        # Convert the Pandas DataFrame to a Numpy array
        # for further computations.
        dictionary['behavioural_data'] = \
                dictionary['behavioural_data'].to_numpy()

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
            duration = behaviour['t'][end_index_behaviour] \
                       - behaviour['t'][start_index_behaviour]
            num_samples = duration * self.sampling_frequency

            dictionary['behavioural_data'] = scipy.signal.resample(
                    dictionary['behavioural_data'], round(num_samples))

        # Create a vector with the correct classification.
        y_len = dictionary['behavioural_data'].shape[0]
        if dictionary['treatment'] == 'None':
            y = torch.ones((y_len, 1), dtype=torch.double)    
        else:
            y = torch.zeros((y_len, 1), dtype=torch.double)    

        return (dictionary['behavioural_data'], y)
