import os
import numpy as np
import glob
from tqdm import tqdm
import pretty_midi as pm
import operator
import math

# For plotting
try: 
    import librosa.display
except ImportError: 
    librosa = None

if librosa is not None:
    def plot_piano_roll(md, start_pitch, end_pitch, fs=100):
        # Use librosa's specshow function for displaying the piano roll
        librosa.display.specshow(md.get_piano_roll(fs)[start_pitch:end_pitch],
                                 hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                                 fmin=pm.note_number_to_hz(start_pitch))

def process_song(filename, time_window = 0.4, max_semitones_allowed = 800):
    '''
    :param filename: Name of the file or path to it, including the name of the file with extension. Ex: Fur Elise.mid
    :param time_window: Time window in seconds (can be float) to sample the song
    :param max_semitones_allowed: Max number of semitones (notes) extracted from all the combined tracks of the song
    :return: List of lists with length of 88. They are all 0 except some that have 1 at the index if that note is
    'on' inside the time window.
    '''
    # Load MIDI file into PrettyMIDI object
    midi_data = pm.PrettyMIDI(filename)
    # midi_data = pm.PrettyMIDI('Moonlight Sonata.mid')
    midi_data.remove_invalid_notes()

    # Get the first 800 semitones of our melody track and delete all the others
    max_semitones = max_semitones_allowed
    for instrument in midi_data.instruments:
        del instrument.notes[max_semitones:]

    # Create new midi object to copy the edited one
    new_midi  = pm.PrettyMIDI()
    # Create piano instrument
    piano = pm.Instrument(program=0, is_drum=False, name='melody')

    new_midi.key_signature_changes = midi_data.key_signature_changes
    new_midi.time_signature_changes = midi_data.time_signature_changes
    new_midi.resolution = midi_data.resolution

    temp_track = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            temp_track.append(note)

    # Sort the combined notes, first by the minor and then by the major attribute
    temp_track.sort(key=operator.attrgetter('end'), reverse=False)
    temp_track.sort(key=operator.attrgetter('start'))

    # Adjust all notes so that the track starts at 0 exactly
    original_start_time = min(temp_track, key=operator.attrgetter('start')).start
    for note in temp_track:
        note.start = note.start - original_start_time
        note.end = note.end - original_start_time
        if any(((x.start == note.start) and (x.end == note.end)) for x in piano.notes): continue
        # Append adjusted notes to new instrument
        piano.notes.append(note)

    # Delete everything after the max number of semitones specified
    del piano.notes[max_semitones:]
    # Add the piano instrument to the PrettyMIDI object
    new_midi.instruments.append(piano)
    print("How many instruments (should be 1)?", len(new_midi.instruments))

    # See what we have in our single track
    # for note in new_midi.instruments[0].notes:
    #     # print("{note}: {start}->{end}".format(note=pm.note_number_to_name(note.pitch),
    #     #                                       start=midi_data.time_to_tick(note.start),
    #     #                                       end=midi_data.time_to_tick(note.end)))
    #     print(note)

    # Write out the MIDI data to a new file
    # new_midi.write('new_midi.mid')

    # Piano range of notes = 21-108
    min_piano_note = 21
    # print(pm.note_name_to_number("A0"))
    # print(pm.note_name_to_number("C8"))

    tw = time_window
    latest_end = math.ceil(max(new_midi.instruments[0].notes, key=operator.attrgetter('end')).end)
    # print("="*20)
    all_samples = []
    new_start = 0
    for i in range(int((latest_end // tw)) + 1):
        sample_tw = [0 for i in range(88)]
        counter = 0
        if (tw * i) > new_midi.instruments[0].notes[-1].end:
            # Our time window is past our last note so we can end
            break
        for note in new_midi.instruments[0].notes[new_start:]:
            counter += 1
            if (tw * i <= note.start < tw * (i + 1)) \
                    or (tw * i <= note.end < tw * (i + 1)):
                # Make our array start with pitches at 0
                sample_tw[note.pitch - min_piano_note] = 1
            else:
                # Since notes are in order, as soon as we find one not inside the time window, we can break
                new_start = new_start + (counter - 1)
                break
        if (sample_tw not in all_samples) or (sum(sample_tw) != 0):
            all_samples.append(sample_tw)

    # See what we stored
    # print(np.argwhere(all_samples))

    return all_samples

# Example Usage
if __name__ == "__main__":
    path = r"D:\Users\Enzo\Downloads\midi\Classical Piano Midis\Beethoven"
    max_files_to_process = 3

    all_songs = []
    for file in tqdm(glob.glob(os.path.join(path, "*.mid"))[:max_files_to_process]):
        str_name = os.path.basename(os.path.normpath(file))
        print("Processing {}".format(str_name))
        song_samples = process_song(file)
        all_songs.append(song_samples)
        print("_"*30)

    print("Done with one folder!")