import numpy as np
import time
import mne
import csv
import os

from pylsl import StreamInfo
from pylsl import StreamOutlet
from pylsl import local_clock

def main():
    basepath = "/home/laphilips/0_workspace/BCI_Dataset/bci-challenge/raw"
    setname = "Data_S02.set"

    raw = mne.io.read_raw_eeglab(os.path.join(basepath, setname), verbose=0)
    ch_names = ["Fp1", "Fp2", "T7", "T8", "O1", "O2", "Fz", "Pz"]
    electrode = raw.ch_names
    picks = [electrode.index(c) for c in ch_names]

    fullSess = raw[picks, :][0]
    fs = raw.info["sfreq"]
    T = 1 / fs
    print(fs, T, fullSess.shape)

    info = StreamInfo('NER2015', 'EEG', 8, 200, 'float32', 'ner2015')
    # append some meta-data
    info.desc().append_child_value("manufacturer", "NER2015")
    channels = info.desc().append_child("channels")
    for c in electrode:
        channels.append_child("channel")\
            .append_child_value("name", c)\
            .append_child_value("unit", "microvolts")\
            .append_child_value("type", "EEG")

    
    # next make an outlet; we set the transmission chunk size to 32 samples and
    # the outgoing buffer size to 360 seconds (max.)
    outlet = StreamOutlet(info, 32, 360)

    print("now sending data...")
    ts, i = 0.001, 400
    while True:
        i = 400 if i == fullSess.shape[1] else i

        sample = fullSess[:, i].tolist()
        outlet.push_sample(sample, timestamp=ts)
        i += 1

        ts += T
        time.sleep(T)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        exit(0)
