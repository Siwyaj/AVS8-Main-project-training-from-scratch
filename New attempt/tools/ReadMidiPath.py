from mido import MidiFile
import numpy as np

def ReadMidi(midi_path):
    """
    Reads the midi file in the given path and returns two dictionaries:
    Midi_event: a dictionary with the midi events and their corresponding time
    Midi_event_time: a dictionary with the midi notes and their corresponding time
    """
    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat

    assert len(midi_file.tracks) == 2, "Midi file should have 2 tracks"
    microseconds_per_beat = midi_file.tracks[0][0].tempo
    beats_per_second = 1e6 / microseconds_per_beat
    ticks_per_second = ticks_per_beat * beats_per_second

    message_list = []
    time_list_sec = []

    ticks = 0
    for message in midi_file.tracks[1]:
        message_list.append(str(message))
        ticks += message.time
        time_list_sec.append(ticks / ticks_per_second)
    
    midi_dict= {
        "midi_event": np.array(message_list),
        "midi_event_time": np.array(time_list_sec)
    }

    return midi_dict


if __name__ == '__main__':
    import os
    testfolder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "testtestfiles"))
    midi_path = os.path.join(testfolder, "TestMidi.midi")
    print("Resolved path:", midi_path)
    midi_dict = ReadMidi(midi_path)
    #print(midi_dict["midi_event"])
    #print(midi_dict["midi_event_time"])

    #write to a txt file in testtest folder
    with open(os.path.join(testfolder, "midi_dict.txt"), "w") as f:
        for i in range(len(midi_dict["midi_event"])):
            f.write(f"{midi_dict['midi_event'][i]} {midi_dict['midi_event_time'][i]}\n")