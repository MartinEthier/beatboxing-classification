from pathlib import Path
import argparse

import yaml
from tqdm import tqdm
import wandb
import torch
import torchvision
from torchvision.transforms import Compose

from dataset import BeatboxDataset
import transforms as tf
from constants import *


def main(config):
    # Get wandb setup
    run = wandb.init(project="beatbox-classification", config=config, entity="methier")
    
    # Access all hyperparameter values through wandb.config for sweep
    cfg = wandb.config

    # Prep checkpoint directory in repo folder
    repo_dir = Path(__file__).resolve().parent
    checkpoint_dir = repo_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Optimizes training speed
    torch.backends.cudnn.benchmark = True
    
    # Prep transforms
    train_transforms = Compose([
        tf.PadToMax(MAX_LENGTH),
        tf.MelSpectrogram(SAMPLE_RATE, n_fft=cfg['n_fft'], hop_length=cfg['hop_length'], n_mels=cfg['n_mels']),
        tf.ToTensor()
    ])
    val_transforms = train_transforms
    
    # Create datasets and data loaders
    root_path = Path(cfg["root"])
    train_set = BeatboxDataset(root_path, "train", CLASS_MAP, train_transforms)
    val_set = BeatboxDataset(root_path, "val", CLASS_MAP, val_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg['batch_size'],
        pin_memory=True,
        num_workers=cfg['num_workers']
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg['batch_size'],
        pin_memory=True,
        num_workers=cfg['num_workers']
    )

    # Init model
    model = getattr(torchvision.models, cfg["model_name"])(pretrained=cfg["pretrained"])
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_MAP))
    model = model.to(device)
    
    # Init optimizer, loss, and LR scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'], eps=cfg['eps'])
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    best_acc = -1.0
    
    for epoch in range(cfg['epochs']):
        # Training loop
        print(f"\n=== Epoch {epoch + 1} ===")
        running_train_loss = 0.0
        train_correct = 0
        model.train()
        
        for samples in tqdm(train_loader, desc='Training'):
            inputs = samples[0].to(device)
            labels = samples[1].to(device)

            # Forward pass
            output = model(inputs)
            preds = torch.argmax(output, dim=1)
            
            # Compute loss with sum reduction for logging
            loss_sum = criterion(output, labels)
            
            # Used for calculating loss and accuracy over epoch
            running_train_loss += loss_sum.item()
            train_correct += (preds == labels).sum().item()

            # Compute actual loss for this batch
            loss = loss_sum / samples[1].shape[0]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation loop
        running_val_loss = 0.0
        val_correct = 0
        preds_all = []
        labels_all = []
        model.eval()
        with torch.no_grad():
            for samples in tqdm(val_loader, desc='Validation'):
                inputs = samples[0].to(device)
                labels = samples[1].to(device)

                # Forward pass
                output = model(inputs)
                preds = torch.argmax(output, dim=1)

                # Collect preds/labels for confusion matrix
                preds_all.append(preds)
                labels_all.append(labels)

                # Compute loss with sum reduction for logging
                loss_sum = criterion(output, labels)

                # Used for calculating loss and accuracy over epoch
                running_val_loss += loss_sum.item()
                val_correct += (preds == labels).sum().item()

        # Needed to generate confusion matrix
        preds_all = torch.cat(preds_all).detach().cpu().numpy()
        labels_all = torch.cat(labels_all).detach().cpu().numpy()

        # Save model if better than current best
        val_acc = val_correct / len(val_set) * 100.0
        if val_acc > best_acc:
            best_acc = val_acc
            wandb.run.summary["best_acc"] = best_acc
            torch.save(model.state_dict(), checkpoint_dir / f"{run.name}_best_model.pt")

            # Only log confusion matrix for best epoch since you can't look at history
            wandb.log({'conf_mat': wandb.plot.confusion_matrix(probs=None, y_true=labels_all, preds=preds_all, class_names=CLASS_MAP)}, commit=False)

        wandb.log({
            'train_loss': running_train_loss / len(train_loader),
            'val_loss': running_val_loss / len(val_loader),
            'train_acc': train_correct / len(train_set) * 100.0,
            'val_acc': val_acc
        })
   
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_name', type=str, help='Name of the config file located in the repo.')
    args = parser.parse_args()
    
    # Load in config
    cfg_path = Path(__file__).parent.absolute() / args.cfg_name
    with cfg_path.open('r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg)
    
    main(cfg)
