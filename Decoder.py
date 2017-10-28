#import pickle

from music21 import midi, converter

from music21.analysis.floatingKey import FloatingKeyException
from tqdm import tqdm

import numpy as np
from music21 import corpus, converter, stream, note, duration, analysis, interval
SLUR_SYMBOL = '__'
START_SYMBOL = 'START'
END_SYMBOL = 'END'
SUBDIVISION = 4
output_file = 'example.mid'

def standard_note(note_or_rest_string):
    if note_or_rest_string == 'rest':
        return note.Rest()
    # treat other additional symbols as rests
    if note_or_rest_string == START_SYMBOL or note_or_rest_string == END_SYMBOL:
        return note.Rest()
    if note_or_rest_string == SLUR_SYMBOL:
        print('Warning: SLUR_SYMBOL used in standard_note')
        return note.Rest()
    else:
        return note.Note(note_or_rest_string)


def indexed_chorale_to_score(seq):
    """

    :param seq: voice major
    :param pickled_dataset:
    :return:
    """
    #_, _, _, index2notes, note2indexes, _ = pickle.load(open(pickled_dataset, 'rb'))
    index2notes = [{0: 'B4', 1: 'D-5', 2: 'F#4', 3: '__', 4: 'G-4', 5: 'C5', 6: 'E-4',
        7: 'A-4', 8: 'rest', 9: 'A#4', 10: 'G5', 11: 'G4', 12: 'D#4', 13: 'F5', 14: 'C#5',
        15: 'B-4', 16: 'A4', 17: 'F4', 18: 'E#4', 19: 'C#4', 20: 'END', 21: 'E5', 22: 'A-5',
        23: 'F#5', 24: 'E#5', 25: 'E-5', 26: 'G-5', 27: 'A5', 28: 'START', 29: 'D4', 30: 'D5',
        31: 'E4', 32: 'G#4', 33: 'C4', 34: 'D#5', 35: 'B#4', 36: 'C-5', 37: 'G#5', 38: 'F-4',
        39: 'D-4', 40: 'F-5', 41: 'B--4', 42: 'B#3', 43: 'F##4', 44: 'C##4', 45: 'G##4', 
        46: 'C##5', 47: 'D##4', 48: 'E--5', 49: 'B--5', 50: 'F##5', 51: 'D##5'}, 
        {0: 'A3', 1: 'B4', 2: 'F#4', 3: 'D-5', 4: '__', 5: 'A-3', 6: 'D-4', 7: 'F3', 
        8: 'G-4', 9: 'C5', 10: 'E-4', 11: 'A-4', 12: 'rest', 13: 'A#4', 14: 'D#4', 
        15: 'G4', 16: 'A#3', 17: 'C#5', 18: 'B-4', 19: 'G#3', 20: 'A4', 21: 'F4', 22: 'E#4',
        23: 'C#4', 24: 'END', 25: 'B-3', 26: 'F#3', 27: 'START', 28: 'B3', 29: 'D4', 30: 'E4',
        31: 'D5', 32: 'F##4', 33: 'G#4', 34: 'C4', 35: 'G3', 36: 'F-4', 37: 'B#3', 38:'B#4',
        39: 'C-4', 40: 'C##4', 41: 'D##4', 42: 'F##3', 43: 'E#3', 44: 'G##3', 45: 'G##4', 
        46: 'A##3', 47: 'G-3', 48: 'C-5', 49: 'B--4', 50: 'E--4', 51: 'A--4', 52: 'E##4', 
        53: 'B--3'}, {0: 'A3', 1: 'F#4', 2: '__', 3: 'D3', 4: 'F3', 5: 'A-3', 6: 'D-4', 7: 'B#3',
        8: 'C-4', 9: 'G-4', 10: 'E-4', 11: 'A-4', 12: 'rest', 13: 'D#4', 14: 'G4', 15: 'C#3',
        16: 'A#3', 17: 'G#3', 18: 'A4', 19: 'F4', 20: 'E#4', 21: 'C#4', 22: 'END', 23: 'B-3',
        24: 'E3', 25: 'F#3', 26: 'START', 27: 'B3', 28: 'D4', 29: 'C3', 30: 'E4', 31: 'D#3',
        32: 'G#4', 33: 'C4', 34: 'G3', 35: 'E-3', 36: 'E#3', 37: 'G-3', 38: 'B--3', 39: 'F-3',
        40: 'F-4', 41: 'C##4', 42: 'B#2', 43: 'D##3', 44: 'F##3', 45: 'C##3', 46: 'G##3', 47: 'D-3',
        48: 'D##4', 49: 'F##4', 50: 'A##3', 51: 'B--4', 52: 'E--4', 53: 'E##3'}, {0: 'F2', 1: 'A3',
        2: '__', 3: 'D3', 4: 'F3', 5: 'A-3', 6: 'G-3', 7: 'D-4', 8: 'B#3', 9: 'D#2', 10: 'C-3',
        11: 'E-4', 12: 'rest', 13: 'G-2', 14: 'E2', 15: 'D#4', 16: 'C#3', 17: 'A2', 18: 'F#2',
        19: 'A#3', 20: 'D2', 21: 'E-2', 22: 'G2', 23: 'G#3', 24: 'C2', 25: 'C#2', 26: 'C#4',
        27: 'E#2', 28: 'END', 29: 'A#2', 30: 'G#2', 31: 'B-3', 32: 'D-3', 33: 'E3', 34: 'B#2',
        35: 'F##3', 36: 'F#3', 37: 'START', 38: 'B3', 39: 'C3', 40: 'D4', 41: 'E4', 42: 'B-2',
        43: 'D#3', 44: 'C4', 45: 'G3', 46: 'A-2', 47: 'E-3', 48: 'E#3', 49: 'B2', 50: 'C-4', 
        51: 'B--3', 52: 'F-3', 53: 'B--2', 54: 'C##3', 55: 'F##2', 56: 'D##2', 57: 'G##2', 
        58: 'C##2', 59: 'D##3', 60: 'B#1', 61: 'G##3', 62: 'D-2', 63: 'A##2', 64: 'E##2', 65: 'F-4',
        66: 'F-2', 67: 'E--3', 68: 'E--4', 69: 'A--3', 70: 'C##4', 71: 'E##3', 72: 'A##3'}]


    note2indexes = [{'B4': 0, 'D-5': 1, 'F#4': 2, '__': 3, 'G-4': 4, 'C5': 5, 'E-4': 6,
        'A-4': 7, 'rest': 8, 'A#4': 9, 'G5': 10, 'G4': 11, 'D#4': 12, 'F5': 13, 'C#5': 14,
        'B-4': 15, 'A4': 16, 'F4': 17, 'E#4': 18, 'C#4': 19, 'END': 20, 'E5': 21, 'A-5': 22,
        'F#5': 23, 'E#5': 24, 'E-5': 25, 'G-5': 26, 'A5': 27, 'START': 28, 'D4': 29, 'D5': 30,
        'E4': 31, 'G#4': 32, 'C4': 33, 'D#5': 34, 'B#4': 35, 'C-5': 36, 'G#5': 37, 'F-4': 38,
        'D-4': 39, 'F-5': 40, 'B--4': 41, 'B#3': 42, 'F##4': 43, 'C##4': 44, 'G##4': 45, 
        'C##5': 46, 'D##4': 47, 'E--5': 48, 'B--5': 49, 'F##5': 50, 'D##5': 51}, {'A3': 0,
        'B4': 1, 'F#4': 2, 'D-5': 3, '__': 4, 'A-3': 5, 'D-4': 6, 'F3': 7, 'G-4': 8, 'C5': 9,
        'E-4': 10, 'A-4': 11, 'rest': 12, 'A#4': 13, 'D#4': 14, 'G4': 15, 'A#3': 16, 'C#5': 17,
        'B-4': 18, 'G#3': 19, 'A4': 20, 'F4': 21, 'E#4': 22, 'C#4': 23, 'END': 24, 'B-3': 25,
        'F#3': 26, 'START': 27, 'B3': 28, 'D4': 29, 'E4': 30, 'D5': 31, 'F##4': 32, 'G#4': 33,
        'C4': 34, 'G3': 35, 'F-4': 36, 'B#3': 37, 'B#4': 38, 'C-4': 39, 'C##4': 40, 'D##4': 41,
        'F##3': 42, 'E#3': 43, 'G##3': 44, 'G##4': 45, 'A##3': 46, 'G-3': 47, 'C-5': 48,
         'B--4': 49, 'E--4': 50, 'A--4': 51, 'E##4': 52, 'B--3': 53}, {'A3': 0, 'F#4': 1, 
         '__': 2, 'D3': 3, 'F3': 4, 'A-3': 5, 'D-4': 6, 'B#3': 7, 'C-4': 8, 'G-4': 9, 
         'E-4': 10, 'A-4': 11, 'rest': 12, 'D#4': 13, 'G4': 14, 'C#3': 15, 'A#3': 16, 
         'G#3': 17, 'A4': 18, 'F4': 19, 'E#4': 20, 'C#4': 21, 'END': 22, 'B-3': 23, 
         'E3': 24, 'F#3': 25, 'START': 26, 'B3': 27, 'D4': 28, 'C3': 29, 'E4': 30, 
         'D#3': 31, 'G#4': 32, 'C4': 33, 'G3': 34, 'E-3': 35, 'E#3': 36, 'G-3': 37, 
         'B--3': 38, 'F-3': 39, 'F-4': 40, 'C##4': 41, 'B#2': 42, 'D##3': 43, 'F##3': 44,
         'C##3': 45, 'G##3': 46, 'D-3': 47, 'D##4': 48, 'F##4': 49, 'A##3': 50, 'B--4': 51,
         'E--4': 52, 'E##3': 53}, {'F2': 0, 'A3': 1, '__': 2, 'D3': 3, 'F3': 4, 'A-3': 5,
         'G-3': 6, 'D-4': 7, 'B#3': 8, 'D#2': 9, 'C-3': 10, 'E-4': 11, 'rest': 12, 'G-2': 13,
         'E2': 14, 'D#4': 15, 'C#3': 16, 'A2': 17, 'F#2': 18, 'A#3': 19, 'D2': 20, 'E-2': 21,
         'G2': 22, 'G#3': 23, 'C2': 24, 'C#2': 25, 'C#4': 26, 'E#2': 27, 'END': 28, 'A#2': 29,
         'G#2': 30, 'B-3': 31, 'D-3': 32, 'E3': 33, 'B#2': 34, 'F##3': 35, 'F#3': 36, 'START': 37,
         'B3': 38, 'C3': 39, 'D4': 40, 'E4': 41, 'B-2': 42, 'D#3': 43, 'C4': 44, 'G3': 45, 
         'A-2': 46, 'E-3': 47, 'E#3': 48, 'B2': 49, 'C-4': 50, 'B--3': 51, 'F-3': 52, 'B--2': 53,
         'C##3': 54, 'F##2': 55, 'D##2': 56, 'G##2': 57, 'C##2': 58, 'D##3': 59, 'B#1': 60, 'G##3': 61,
         'D-2': 62, 'A##2': 63, 'E##2': 64, 'F-4': 65, 'F-2': 66, 'E--3': 67, 'E--4': 68, 'A--3': 69, 
         'C##4': 70, 'E##3': 71, 'A##3': 72}]
    num_pitches = list(map(len, index2notes))
    slur_indexes = list(map(lambda d: d[SLUR_SYMBOL], note2indexes))

    score = stream.Score()
    for voice_index, v in enumerate(seq):
        part = stream.Part(id='part' + str(voice_index))
        dur = 0
        f = note.Rest()
        for k, n in enumerate(v):
            # if it is a played note
            if not n == slur_indexes[voice_index]:
                # add previous note
                if dur > 0:
                    f.duration = duration.Duration(dur / SUBDIVISION)
                    part.append(f)

                dur = 1
                f = standard_note(index2notes[voice_index][n])
            else:
                dur += 1
        # add last note
        f.duration = duration.Duration(dur / SUBDIVISION)
        part.append(f)
        score.insert(part)
    return score

def main():

    #pickled_dataset = 'bach_dataset.pickle'
    #pickled_dataset = '/home/ywh/SeqGAN-master/bach_dataset.pickle'
    seq1 = []
    for i in range(0,4):
        positive_examples = []
        a = []
        with open('test_data/eval_file%d.txt'%i)as fin:
              for line in fin:
                  line = line.strip()
                  line = line.split()
                  parse_line = [round(float(x)) for x in line]
                  positive_examples.append(parse_line)
              for t in range(0,1):
                  a.extend(positive_examples[t])
        #print(a)
        seq1.append(a)
        #print(seq1)


    seq = np.transpose(seq1)
    print(len(seq))

    score = indexed_chorale_to_score(np.transpose(seq, axes=(1, 0)))

    # save as MIDI file
    mf = midi.translate.music21ObjectToMidiFile(score)
    mf.open(output_file, 'wb')
    mf.write()
    mf.close()
    print("File " + output_file + " written")


if __name__ == '__main__':
    main()