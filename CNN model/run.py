from DataLoader import PianoTranscriptionDataset
import os
import torch

import torch.optim as optim
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from CNNmodel import CRNNTranscriptionModel  # Your model class
import os
import torch.nn as nn

bce = nn.BCELoss()
mse = nn.MSELoss()

def compute_loss(outputs, targets):
    loss_onset = bce(outputs['onset'], targets['onset'])
    loss_offset = bce(outputs['offset'], targets['offset'])
    loss_frame = bce(outputs['frame'], targets['frame'])
    loss_velocity = mse(outputs['velocity'], targets['velocity'])
    loss_reg_onset = mse(outputs['regression_onset'], targets['onset'])  # regression-onset vs. true onsets

    total_loss = (
        loss_onset +
        loss_offset +
        loss_frame +
        loss_velocity +
        loss_reg_onset
    )
    return total_loss


def train(model, dataloader, device):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)

    model.train()
    step = 0

    for epoch in range(1, 5):  # Adjust number of epochs as needed
        for mel_batch, target_batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            mel_batch = mel_batch.to(device)  # (B, T, mel_bins)

            # Move all target tensors to the correct device
            targets = {k: v.to(device) for k, v in target_batch.items()}

            # Forward pass
            outputs = model(mel_batch)

            # Compute loss
            loss = compute_loss(outputs, targets)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")



if __name__ == "__main__":
    print("Starting program...")
    import torch

    print("preparing dataset...")
    dataset = PianoTranscriptionDataset(
        audio_dir='maestro-v3.0.0/reorginized/wavs',
        midi_dir='maestro-v3.0.0/reorginized/midis',
        sample_rate=16000,
        duration=10.0
    )
    print("dataset prepared")

    print("preparing dataloader...")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    print("dataloader prepared")

    print("preparing model...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("preparing model...")
    model = CRNNTranscriptionModel()  # Your model class
    print("model prepared")

    print("training model...")
    train(model, dataloader, device)

