from midi2audio import FluidSynth

# Convert MIDI to WAV
def midi_to_wav(midi_file, wav_file):
    fs = FluidSynth()
    fs.midi_to_audio(midi_file, wav_file)
    print(f"Conversion to WAV complete: {wav_file}")

# Convert WAV to MP3
def wav_to_mp3(wav_file, mp3_file):
    audio = AudioSegment.from_wav(wav_file)
    audio.export(mp3_file, format="mp3")
    print(f"Conversion to MP3 complete: {mp3_file}")

# Example usage
midi_file = 'testMidi.midi'  # Path to your MIDI file
wav_file = 'output_file.wav'  # Output WAV file

# Convert MIDI to WAV
midi_to_wav(midi_file, wav_file)
