"""
This file contains the configuration for data loading.
Actual values are just copies for now. They will be updated later. as well as remove this line
"""

sample_rate = 16000
classes_num = 88    # Number of notes of piano
begin_note = 21     # MIDI note of A0, the lowest note of a piano.
segment_seconds = 10.	# Training segment duration
hop_seconds = 1.
frames_per_second = 100
velocity_scale = 128