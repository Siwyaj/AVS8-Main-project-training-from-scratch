import torch
import torch.nn as nn
import torch.nn.functional as F
import torchlibrosa

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 1))  # Pool only over frequency
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        return x

class CRNNTranscriptionModel(nn.Module):
    def __init__(self, n_mels=229, dropout_conv=0.2, dropout_fc=0.5):
        super().__init__()
        # Replace with this corrected line in CNNmodel.py
        self.spectrogram_extractor = torchlibrosa.stft.LogmelFilterBank(
            sr=16000, 
            n_fft=1024, 
            n_mels=n_mels, 
            fmin=30.0, 
            fmax=8000.0
        )

        self.bn_input = nn.BatchNorm2d(1)

        self.conv_blocks = nn.Sequential(
            ConvBlock(1, 48, dropout_conv),
            ConvBlock(48, 64, dropout_conv),
            ConvBlock(64, 92, dropout_conv),
            ConvBlock(92, 128, dropout_conv)
        )

        self.flatten_fc = nn.Linear(128 * (n_mels // (2**4)), 768)

        self.bi_gru1 = nn.GRU(768, 256, num_layers=1, batch_first=True, bidirectional=True)
        self.bi_gru2 = nn.GRU(512, 256, num_layers=1, batch_first=True, bidirectional=True)

        # Prediction heads (each is (B, T, 88))
        self.onset_fc = nn.Sequential(nn.Dropout(dropout_fc), nn.Linear(512, 88), nn.Sigmoid())
        self.offset_fc = nn.Sequential(nn.Dropout(dropout_fc), nn.Linear(512, 88), nn.Sigmoid())
        self.frame_fc = nn.Sequential(nn.Dropout(dropout_fc), nn.Linear(512, 88), nn.Sigmoid())
        self.velocity_fc = nn.Sequential(nn.Dropout(dropout_fc), nn.Linear(512, 88), nn.Sigmoid())

        # Regression-onset stream
        self.reg_gru = nn.GRU(88 * 2, 256, num_layers=1, batch_first=True, bidirectional=True)
        self.reg_fc = nn.Sequential(nn.Dropout(dropout_fc), nn.Linear(512, 88), nn.Sigmoid())

    def forward(self, x):
        # x: (B, T)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (B, 1, T), adding channel dimension if not present

        # Extract mel spectrogram (output: (B, n_mels, T))
        mel = self.spectrogram_extractor(x)  
        mel = self.bn_input(mel)

        # Convolutional layers (output: (B, C, mel', T))
        x = self.conv_blocks(mel)  

        # Reshaping for further processing
        x = x.permute(0, 3, 1, 2)  # (B, T, C, mel')
        B, T, C, F = x.shape
        x = x.reshape(B, T, C * F)  # Flatten (B, T, features)

        # Fully connected layer
        x = self.flatten_fc(x)  # (B, T, 768)

        # BiGRU layers
        x, _ = self.bi_gru1(x)
        x, _ = self.bi_gru2(x)

        # Output layers for different tasks
        onset = self.onset_fc(x)
        offset = self.offset_fc(x)
        frame = self.frame_fc(x)
        velocity = self.velocity_fc(x)

        # Regression-onset head (uses onset + velocity)
        reg_input = torch.cat([onset, velocity], dim=-1)
        reg_out, _ = self.reg_gru(reg_input)
        regression_onset = self.reg_fc(reg_out)

        return {
            'onset': onset,
            'offset': offset,
            'frame': frame,
            'velocity': velocity,
            'regression_onset': regression_onset
        }

