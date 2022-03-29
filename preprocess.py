from pathlib import Path
from itertools import chain
import argparse
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import librosa

from constants import FILE_TRIMS


file_types = ('Snare', 'Kick', 'HHopened', 'HHclosed', 'Improvisation')


def main(root_path, val_ratio):
    # Create folder to save preprocessed data to
    processed_path = root_path / "processed_dataset"
    processed_path.mkdir()

    # Combine personal and fixed subsets into a single generator
    personal_path = root_path / "Personal"
    fixed_path = root_path / "Fixed"
    combined_glob = chain(personal_path.glob("[!.]*"), fixed_path.glob("[!.]*"))

    # For generating final plots
    sample_sizes = []
    sample_rate = 0
    accumulator = np.zeros(100000, dtype=np.float32)

    print(f"Processing audio files and saving to {processed_path}...")
    dataset_paths = []
    sample_ID = 0
    for participant_path in combined_glob:
        # Get participant number and subset (Fixed or Personal)
        pnum = participant_path.stem.split('_')[-1]
        subset = participant_path.parent.stem

        # Get wav+csv files for each audio clip under this participant
        for file_type in file_types:
            filename = f"P{pnum}_{file_type}_{subset}"

            # Load in onset and class labels from csv
            csv_path = participant_path / f"{filename}.csv"
            label_df = pd.read_csv(csv_path, header=None)
            #print(label_df)
            onset_labels = label_df.iloc[:, 0].to_numpy()
            class_labels = label_df.iloc[:, 1].tolist()
            assert onset_labels.shape[0] == len(class_labels)

            # Load waveform
            wav_path = participant_path / f"{filename}.wav"
            waveform, sample_rate = librosa.load(wav_path, sr=None)

            # Trim silent end of waveform if indicated in FILE_TRIMS
            for trim_obj in FILE_TRIMS:
                if subset == trim_obj[0] and pnum == trim_obj[1] and file_type == trim_obj[2]:
                    trim_idx = librosa.time_to_samples(trim_obj[3], sr=sample_rate)
                    waveform = waveform[:trim_idx]

            # Split up original waveform at onset indices
            onset_idx = librosa.time_to_samples(onset_labels, sr=sample_rate)
            split_waveforms = np.split(waveform, onset_idx)[1:]

            # Save each individual sample
            for wf, cls in zip(split_waveforms, class_labels):
                # Trim silent ends off of the sample waveform
                trimmed_wf, _ = librosa.effects.trim(wf)

                # Some samples are messed up and super short, just skip them
                if trimmed_wf.shape[0]/sample_rate < 0.01:
                    continue
                    
                # Some samples have no class label
                if cls.strip() == '':
                    continue

                # Save sample array with class label in filename (zero pad sample ID to 5 digits)
                sample_path = processed_path / f"{sample_ID:05}_{cls.strip()}.npy"
                np.save(sample_path, trimmed_wf)
                sample_ID += 1
                dataset_paths.append(sample_path.name)

                # Keep track of sample sizes for histogram
                sample_sizes.append(trimmed_wf.shape[0]/sample_rate)

                # Zero pad waveform and add to accumulator to generate final waveform plot for full dataset
                padded_wf = np.pad(trimmed_wf, (0, accumulator.shape[0] - trimmed_wf.shape[0] % accumulator.shape[0]), 'constant')
                accumulator += padded_wf

            
    # Split dataset into train/val and save list
    print("Generating train/val lists...")
    random.Random(0).shuffle(dataset_paths)
    val_idx = int(val_ratio * len(dataset_paths))

    val_paths = dataset_paths[:val_idx]
    val_list_path = processed_path / "val_list.txt"
    with val_list_path.open('w') as f:
        for p in val_paths:
            f.write(f"{p}\n")

    train_paths = dataset_paths[val_idx:]
    train_list_path = processed_path / "train_list.txt"
    with train_list_path.open('w') as f:
        for p in train_paths:
            f.write(f"{p}\n")
    
    # Generate plots to help figure out best max length
    print("Displaying dataset plots...")
    plt.hist(sample_sizes, bins='auto')
    plt.title("Histogram of sample lengths")
    plt.xlabel("Time (s)")
    plt.show()
    time_axis = np.arange(0, accumulator.shape[0]) / sample_rate
    plt.plot(time_axis, accumulator, linewidth=1)
    plt.title("All sample waveforms accumulated")
    plt.xlabel("Time (s)")
    plt.show()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', type=str, required=True, help='')
    parser.add_argument('--val_ratio', '-v', type=float, default=0.2, help='')
    args = parser.parse_args()

    root_path = Path(args.root)

    main(root_path, args.val_ratio)
