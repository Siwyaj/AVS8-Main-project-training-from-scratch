import tensorflow as tf
from tensorflow.keras import layers, models

def conv_block(x, filters):
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D(pool_size=(2, 1))(x)  # only downsample frequency
    return x

def build_crnn(input_shape=(128, None, 1), rnn_units=256):
    inputs = tf.keras.Input(shape=input_shape)  # (freq, time, channels)
    
    # Apply batch norm to input across freq bins
    x = layers.BatchNormalization()(inputs)

    # 4 convolutional blocks
    x = conv_block(x, 48)
    x = conv_block(x, 64)
    x = conv_block(x, 92)
    x = conv_block(x, 128)

    # Reshape for RNN: (batch, time, features)
    x = layers.Permute((2, 1, 3))(x)  # (batch, time, freq, channels)
    x = layers.Reshape((-1, x.shape[2] * x.shape[3]))(x)  # (batch, time, features)

    # 2 Bidirectional GRU layers
    x = layers.Bidirectional(layers.GRU(rnn_units, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(rnn_units, return_sequences=True))(x)

    # Output heads (128 values each time step)
    onset_out = layers.Dense(128, activation='sigmoid', name='onset')(x)
    offset_out = layers.Dense(128, activation='sigmoid', name='offset')(x)
    frame_out = layers.Dense(128, activation='sigmoid', name='frame')(x)
    velocity_out = layers.Dense(128, activation='linear', name='velocity')(x)

    return models.Model(inputs=inputs, outputs={
        'onset': onset_out,
        'offset': offset_out,
        'frame': frame_out,
        'velocity': velocity_out,
    })

model = build_crnn()
model.compile(
    optimizer='adam',
    loss={
        'onset': 'binary_crossentropy',
        'offset': 'binary_crossentropy',
        'frame': 'binary_crossentropy',
        'velocity': 'mse',
    },
    loss_weights={
        'onset': 1.0,
        'offset': 1.0,
        'frame': 1.0,
        'velocity': 0.5,  # Tune this if velocity isn't as important
    },
    metrics=['accuracy']
)
model.summary()


import numpy as np
import librosa
from librosa.feature import melspectrogram

import pretty_midi

def wav_to_logmel(wav_path, sr=16000, n_mels=128, hop_length=160, win_length=512):
    y, _ = librosa.load(wav_path, sr=sr)
    mel = melspectrogram(y=y, sr=sr, n_mels=n_mels,
                     hop_length=hop_length, win_length=win_length)

    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = logmel.T  # shape: [time, mel]
    return logmel

def midi_to_targets(midi_path, n_frames, fps=100):
    midi = pretty_midi.PrettyMIDI(midi_path)
    onset = np.zeros((n_frames, 128), dtype=np.float32)
    offset = np.zeros((n_frames, 128), dtype=np.float32)
    frame = np.zeros((n_frames, 128), dtype=np.float32)
    velocity = np.zeros((n_frames, 128), dtype=np.float32)

    for note in midi.instruments[0].notes:
        onset_frame = int(note.start * fps)
        offset_frame = int(note.end * fps)
        velocity_value = note.velocity / 127.0  # normalize

        if onset_frame < n_frames:
            onset[onset_frame, note.pitch] = 1
            velocity[onset_frame, note.pitch] = velocity_value

        for f in range(onset_frame, min(offset_frame + 1, n_frames)):
            frame[f, note.pitch] = 1

        if offset_frame < n_frames:
            offset[offset_frame, note.pitch] = 1

    return onset, offset, frame, velocity

if __name__ == "__main__":
    # Paths to your test files
    test_wav = "testWav.wav"
    test_midi = "testMidi.midi"

    # Preprocess audio
    logmel = wav_to_logmel(test_wav)  # shape: [T, 128]
    logmel = logmel[:, :, np.newaxis]  # [T, 128, 1]
    logmel = np.transpose(logmel, (1, 0, 2))  # [128, T, 1]
    logmel = np.expand_dims(logmel, axis=0)  # [1, 128, T, 1]

    # Frame count
    n_frames = logmel.shape[2]

    # Preprocess MIDI
    onset, offset, frame, velocity = midi_to_targets(test_midi, n_frames)

    # Build and run model
    model = build_crnn()
    preds = model.predict(logmel)

    # Print output shapes
    print("Predictions:")
    for key in preds:
        print(f"{key}: {preds[key].shape}")

    # Optional: Compare prediction to ground truth
    print("\nGround truth example (frame):", frame.shape)
    print("First frame onset (ground truth):", onset[0])
    print("First frame onset (prediction):", preds["onset"][0][0])
