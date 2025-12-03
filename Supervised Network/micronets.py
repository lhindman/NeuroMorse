# %%
# ================================================
# Title: Multi-Trial Spike Dataset Visualization
# ================================================

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# %%
# ================================================
# Dataset Definition
# ================================================

class NeuroMorseDataset(Dataset):
    """
    Expects HDF5 structure:
    /Spikes/Channel0 : list of spike time arrays (microseconds)
    /Spikes/Channel1 : list of spike time arrays (microseconds)
    /Labels/Labels   : list of integer ground-truth labels
    """
    
    symbol_space = 5 # time steps between symbols (dot/dash) in a character
    character_space = 10 # time steps between characters in a word
    word_space = 15 # time steps between words
    

    def __init__(self, h5_path, dt_us=1000, test_mode=False):
        super().__init__()
        self.h5_path = h5_path
        self.dt_us = dt_us

        with h5py.File(h5_path, "r") as f:
            self.channels = f["Spikes/Channels"][:]
            self.times = f["Spikes/Times"][:]
            self.labels = f["Labels/Labels"][:]
            if test_mode:
                self.start_times = f["Labels/Start Times"][:]
                self.end_times = f["Labels/End Times"][:]

        self.num_trials = len(self.channels)

        # Create integer labels for classification
        self.labels = np.array([lbl.decode('utf-8') if isinstance(lbl, bytes) else str(lbl) for lbl in self.labels])
        self.label_map = {idx: label for idx, label in enumerate(self.labels)}


    def __len__(self):
        return self.num_trials

    def __getitem__(self, idx):
        string_label = self.labels[idx]
        integer_label = idx  # since integer_labels are unique per trial
        
        # print("Idx:", idx, "String label:", string_label, "-> Integer label:", integer_label)

        spike_times = np.array(self.times[idx]).astype(int)
        spike_channels = np.array(self.channels[idx]).astype(int)
        
        spikes_c0 = spike_times[spike_channels == 0]
        spikes_c1 = spike_times[spike_channels == 1]
        
        # spike_train = np.zeros((int(data[-1][0])+1+word_space,2)) # allocate 2 dimensional tensor with length of last spike time + 1 + word_space
        
        spike_train = np.zeros((spike_times[-1] + 1 + NeuroMorseDataset.word_space, 2), dtype=np.float32)
        spike_train[(spikes_c0 // self.dt_us).astype(int), 0] = 1.0
        spike_train[(spikes_c1 // self.dt_us).astype(int), 1] = 1.0
        # print("Spike train shape:", spike_train.shape)

        return torch.tensor(spike_train, dtype=torch.float32), integer_label
    
    def get_test_times(self, integer_label):
        """" Returns the start times and end times for a given integer class label. """
        if not hasattr(self, 'start_times') or not hasattr(self, 'end_times'):
            raise AttributeError("Dataset was not initialized in test mode; start and end times are unavailable.")
        
        str_label = self.get_class_name(integer_label)
        start_list = []
        end_list = []
        for i in range(self.num_trials):
            if self.labels[i] == str_label:
                start_list.append(self.start_times[i])
                end_list.append(self.end_times[i])
        return start_list, end_list

    def get_class_name(self, integer_label):
        """" Converts an integer class back to its string representation. """
        return self.label_map.get(integer_label,"Unknown")
    
    def custom_collate_fn(batch):
        
        spike_trains = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        
        # Find the maximum sequence length in the current batch
        max_train_length = max([spike_train.shape[0] for spike_train in spike_trains])
        # print("Max length in batch:", max_train_length)

        # Pad each sequence to the maximum length at the beginning
        padded_spike_trains = []
        for spike_train in spike_trains:
            padding_needed = max_train_length - spike_train.shape[0]
            # Create a padding tensor of zeros (or any desired padding value)
            # print("Padding needed:", padding_needed)
            # print("Spike train shape:", spike_train.shape)
            padding = np.zeros((padding_needed, spike_train.shape[1]))
            # print("Padding shape:", padding.shape)
            # Concatenate the padding at the beginning of the sequence
            # padded_spike_train = np.concatenate((padding, spike_train), axis=1)
            
            # Append the padding at the end of the sequnence
            padded_spike_train = np.concatenate((spike_train, padding), axis=0)
            padded_spike_train = torch.tensor(padded_spike_train, dtype=torch.float32)
            
            padded_spike_trains.append((padded_spike_train))
        # Stack the padded sequences into a single tensor
        padded_spike_trains = torch.stack(padded_spike_trains)
        labels = torch.tensor(labels)
        
        return padded_spike_trains,labels

class MicroNetCaptureDataset(Dataset):
    """
    Expects HDF5 structure:
    /Spikes/ch0_times_us : list of spike time arrays (microseconds)
    /Spikes/ch1_times_us : list of spike time arrays (microseconds)
    """
    def __init__(self, h5_path, duration_us, dt_us=1000):
        super().__init__()
        self.h5_path = h5_path
        self.duration_us = duration_us
        self.dt_us = dt_us

        with h5py.File(h5_path, "r") as f:
            self.channel0_spikes = f["Spikes/ch0_times_us"][:]
            self.channel1_spikes = f["Spikes/ch1_times_us"][:]
            self.labels = f["Labels/Labels"][:]

        self.num_trials = len(self.labels)
        self.time_steps = int(np.ceil(duration_us / dt_us))

        # Create integer labels for classification
        self.labels = np.array([lbl.decode('utf-8') if isinstance(lbl, bytes) else str(lbl) for lbl in self.labels])
        self.label_map = {idx: label for idx, label in enumerate(np.unique(self.labels))}
        self.reverse_label_map = {label: idx for idx, label in enumerate(np.unique(self.labels))}

    def __len__(self):
        return self.num_trials

    def __getitem__(self, idx):
        string_label = self.labels[idx]
        integer_label = self.reverse_label_map[string_label]

        spikes_c0 = np.array(self.channel0_spikes[idx])
        spikes_c1 = np.array(self.channel1_spikes[idx])

        spike_train = np.zeros((2, self.time_steps), dtype=np.float32)
        spike_train[0, (spikes_c0 // self.dt_us).astype(int)] = 1.0
        spike_train[1, (spikes_c1 // self.dt_us).astype(int)] = 1.0
        # print("Spike train shape:", spike_train.shape)

        return torch.tensor(spike_train), integer_label
    
    def get_class_name(self, integer_label):
        """" Converts an integer class back to its string representation. """
        return self.label_map.get(integer_label,"Unknown")




# %%
# ================================================
# Visualization Function
# ================================================print

def plot_spike_trials(dataset, max_trials=10, figsize=(10, 6)):
    """
    Plot spike rasters for multiple trials from the dataset.
    Each trial shows spikes for two channels, labeled by ground-truth class.
    """

    num_trials = min(len(dataset), max_trials)
    fig, axes = plt.subplots(num_trials, 1, figsize=figsize, sharex=True)
    if num_trials == 1:
        axes = [axes]  # ensure iterable

    time_axis = np.arange(0, dataset.time_steps * dataset.dt_us, dataset.dt_us) / dataset.dt_us  # convert µs to ms

    # print("Time axis (ms):", time_axis)

    for idx in range(num_trials):
        spike_train, label = dataset[idx]
        spike_train = spike_train.numpy()
        # print("Spike train for trial", idx, ":", spike_train)
        ax = axes[idx]

        # print("Label for trial", idx, ":", label)
        # print("String label:", dataset.get_class_name(label))

        # Plot spikes for Channel 1 (offset vertically for clarity)
        t1 = time_axis[spike_train[1] > 0]
        # print("Channel 1 shape:", spike_train[1].shape)
        ax.scatter(t1, np.ones_like(t1), marker="|", color="tab:blue", s=100, label="DASH (Ch1)")

        # Plot spikes for Channel 0
        t0 = time_axis[spike_train[0] > 0]
        # print("Channel 0 shape:", spike_train[0].shape)
        ax.scatter(t0, np.zeros_like(t0), marker="|", color="tab:orange", s=100, label="DOT (Ch0)")



        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Ch0", "Ch1"])
        ax.set_ylabel(f"Trial {idx}\n(Label={dataset.get_class_name(label)})", rotation=0, labelpad=40, va='center')
        ax.grid(True, linestyle="--", alpha=0.3)

        if idx == 0:
            ax.legend(loc="upper right", fontsize="small")

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(f"Multi-Trial Spike Raster ({num_trials} Trials)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()

# # %%
# # ================================================
# # Load Dataset and Display Visualization
# # ================================================

# # Path to your HDF5 file
# # h5_path = "data/Clean-Train.h5"  # <-- change this to your file path
# h5_path = "../neuromorse_capture/tools/capture_20251030_140437.h5"  # <-- change this to your file path

# # Define dataset parameters
# duration_us = 200000  # total duration of each trial in µs (e.g. 200 ms)

# # Load dataset
# # dataset = NeuroMorseDataset(h5_path, duration_us, dt_us=1)
# dataset = MicroNetCaptureDataset(h5_path, duration_us, dt_us=1000)

# # Visualize first few trials
# plot_spike_trials(dataset, max_trials=10, figsize=(12, 8))
