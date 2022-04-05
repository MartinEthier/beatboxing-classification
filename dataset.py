from pathlib import Path

import numpy as np
import torch


class BeatboxDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, processed_ds_path, split, class_map, transforms=None):
        self.root_path = processed_ds_path
        self.split = split
        self.transforms = transforms
        self.class_map = class_map

        # Load list of filenames from the txt file
        dataset_list = self.root_path / f"{split}_list.txt"
        with dataset_list.open('r') as f:
            self.dataset = [self.root_path / line.rstrip() for line in f]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get class ID
        file_path = self.dataset[idx]
        class_name = file_path.stem.split('_')[-1]
        class_id = self.class_map.index(class_name)

        # Load in waveform array
        waveform = np.load(file_path)

        # Apply transforms
        if self.transforms is not None:
            waveform = self.transforms(waveform)

        return (waveform, class_id)


if __name__=="__main__":
    # Testing the dataset class
    from matplotlib import pyplot as plt
    import transforms as tf
    from constants import *
    from torchvision.transforms import Compose

    transforms = Compose([
        tf.PadToMax(MAX_LENGTH),
        tf.MelSpectrogram(SAMPLE_RATE),
        tf.ToTensor()
    ])

    dataset = BeatboxDataset(Path("/home/martin/school/4B/MSCI_446/AVP_Dataset/processed_dataset"), "train", CLASS_MAP, transforms)
    print(len(dataset))
    
    sample = dataset[1]
    plt.imshow(sample[0])
    plt.show()
