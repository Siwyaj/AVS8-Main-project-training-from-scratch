import torch
import torch.nn.functional as F
import os
import torchaudio
import pretty_midi
import torchlibrosa
import numpy as np

def process_midi(midi_path, n_frames, hop_length=320, sample_rate=16000):
    pm = pretty_midi.PrettyMIDI(midi_path)

    # Initialize empty matrices for onset, offset, frame, and velocity
    onset = np.zeros((n_frames, 88), dtype=np.float32)
    offset = np.zeros((n_frames, 88), dtype=np.float32)
    frame = np.zeros((n_frames, 88), dtype=np.float32)
    velocity = np.zeros((n_frames, 88), dtype=np.float32)

    for note in pm.instruments[0].notes:
        pitch = note.pitch - 21  # MIDI pitch to index (21=A0, 108=C8)
        if pitch < 0 or pitch > 87:
            continue

        onset_idx = int(note.start * sample_rate / hop_length)
        offset_idx = int(note.end * sample_rate / hop_length)

        if onset_idx >= n_frames:
            continue
        offset_idx = min(offset_idx, n_frames - 1)

        # Clip or pad
        onset[onset_idx, pitch] = 1.0
        offset[offset_idx, pitch] = 1.0
        frame[onset_idx:offset_idx+1, pitch] = 1.0
        velocity[onset_idx, pitch] = note.velocity / 127.0  # Normalize

    return {
        'onset': torch.tensor(onset),
        'offset': torch.tensor(offset),
        'frame': torch.tensor(frame),
        'velocity': torch.tensor(velocity),
    }

class PianoTranscriptionDataset(torch.utils.data.Dataset):
    def __init__(self, audio_dir, midi_dir, sample_rate=16000, duration=10.0, hop_length=320, n_mels=229):
        self.audio_dir = audio_dir
        self.midi_dir = midi_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.max_len_samples = int(sample_rate * duration)
        self.max_frames = int(self.max_len_samples / hop_length)

        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])

        # Spectrogram and LogMel filterbank extractor
        self.spectrogram_extractor = torchlibrosa.stft.Spectrogram(
            n_fft=1024,
            hop_length=hop_length,
            win_length=1024,
            window="hann",
            center=True,
            pad_mode="reflect",
            power=2.0
        )

        self.logmel_extractor = torchlibrosa.stft.LogmelFilterBank(
            sr=sample_rate,
            n_fft=1024,
            n_mels=n_mels,
            fmin=30.0,
            fmax=8000.0,
            ref=1.0,
            amin=1e-10,
            top_db=None
        )

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        midi_path = os.path.join(self.midi_dir, self.audio_files[idx].replace('.wav', '.midi'))

        # Load the waveform
        waveform, sr = torchaudio.load(audio_path)
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)

        # Split waveform into smaller chunks if it's longer than max_len_samples
        num_chunks = waveform.shape[1] // self.max_len_samples
        chunks = []
        for i in range(num_chunks):
            chunk = waveform[:, i * self.max_len_samples:(i + 1) * self.max_len_samples]
            chunks.append(chunk)

        # Handle the remainder part if there's any leftover portion after splitting
        remainder = waveform.shape[1] % self.max_len_samples
        if remainder > 0:
            # Take the last chunk from the remainder
            remainder_chunk = waveform[:, -remainder:]
            chunks.append(remainder_chunk)

        # Pad chunks to ensure all chunks are of the same size
        padded_chunks = []
        for chunk in chunks:
            if chunk.shape[1] < self.max_len_samples:
                pad_len = self.max_len_samples - chunk.shape[1]
                chunk = F.pad(chunk, (0, pad_len))  # Pad along the time axis
            padded_chunks.append(chunk)

        # Generate Mel spectrograms for each chunk
        mel_chunks = []
        for chunk in padded_chunks:
            spec = self.spectrogram_extractor(chunk)  # (1, F, T)
            mel = self.logmel_extractor(spec)         # (1, mel, T)
            mel = mel.squeeze(0).transpose(0, 1)      # (T, mel)
            mel_chunks.append(mel)

        # Process the corresponding MIDI file to get the target
        target = process_midi(midi_path, mel_chunks[0].shape[0])  # Ensure target length matches mel

        # Pad MIDI targets to match the length of mel spectrograms
        target_length = mel_chunks[0].shape[0]
        target = {
            key: value[:target_length] if value.shape[0] > target_length else F.pad(value, (0, target_length - value.shape[0]))
            for key, value in target.items()
        }

        # Ensure target lengths match mel chunk length
        assert mel_chunks[0].shape[0] == target['onset'].shape[0], \
            f"Mel spectrogram length {mel_chunks[0].shape[0]} doesn't match target length {target['onset'].shape[0]}"

        return mel_chunks, target
