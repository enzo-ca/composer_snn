import pretty_midi as pm
import numpy as np

# For plotting
import librosa.display
import matplotlib.pyplot as plt

def plot_piano_roll(md, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(md.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pm.note_number_to_hz(start_pitch))

# Load MIDI file into PrettyMIDI object
# midi_data = pm.PrettyMIDI('Fur Elise.mid')
midi_data = pm.PrettyMIDI('Moonlight Sonata.mid')
midi_data.remove_invalid_notes()

all_notes = []
for idx, instrument in enumerate(midi_data.instruments):
    notes = []
    for note in instrument.notes:
        if instrument.is_drum:
            notes.append(0)
            break
        notes.append(note.pitch)
    all_notes.append(notes)

# mins = []
maxs = []
for idx, track in enumerate(all_notes):
    print("Track {} Min: {}, Max: {}".format(idx, min(track), max(track)))
    # mins.append(min(track))
    maxs.append(max(track))

# max_min = np.argmax(mins)
max_max = np.argmax(maxs)

# Delete instruments that are not the main melody
indices_to_delete = [i for i in range(len(midi_data.instruments)) if i != max_max]
for idx in sorted(indices_to_delete, reverse=True):
    del midi_data.instruments[idx]

# Delete instruments that are not the main melody
# for idx, instrument in enumerate(midi_data.instruments):
#     if (idx != max_max):
#         del midi_data.instruments[idx]

# Get the first 800 semitones of our melody track and delete all the others
del midi_data.instruments[max_max].notes[10:]

# Create new midi object to copy the edited one
new_midi  = pm.PrettyMIDI()
# Create piano instrument
piano = pm.Instrument(program=0, is_drum=False, name='melody')

new_midi.key_signature_changes = midi_data.key_signature_changes
new_midi.time_signature_changes = midi_data.time_signature_changes
new_midi.resolution = midi_data.resolution

# Adjust all notes so that the track starts at 0 exactly
original_start_time = midi_data.instruments[max_max].notes[0].start
for note in midi_data.instruments[max_max].notes:
    note.start = note.start - original_start_time
    note.end = note.end - original_start_time
    # Append adjusted notes to new instrument
    piano.notes.append(note)

# Add the piano instrument to the PrettyMIDI object
new_midi.instruments.append(piano)

# Store the notes in numpy array format
for note in new_midi.instruments[0].notes:
    print("{note}: {start}->{end}".format(note=pm.note_number_to_name(note.pitch),
                                          start=midi_data.time_to_tick(note.start),
                                          end=midi_data.time_to_tick(note.end)))
    # print(note)

# Write out the MIDI data
new_midi.write('new_midi.mid')