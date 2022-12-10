"""Example program to demonstrate how to send a multi-channel time-series
with proper meta-data to LSL."""
import pandas as pd
import numpy as np
import random
import time
import csv
import os

from pylsl import StreamInfo
from pylsl import StreamOutlet
from pylsl import local_clock

def main():
    # first create a new stream info (here we set the name to BioSemi,
    # the content-type to EEG, 8 channels, 100 Hz, and float-valued data) The
    # last value would be the serial number of the device or some other more or
    # less locally unique identifier for the stream as far as available (you
    # could also omit it but interrupted connections wouldn't auto-recover).

    fd = open("102022_215547.csv", newline='')
    rows = list(csv.reader(fd)) # 8 ch, 1000hz
    fd.close()
    electrode = rows[10][2:2+8]

    info = StreamInfo('BioSemi', 'EEG', 8, 1000, 'float32', 'biosemi0001')
    # append some meta-data
    info.desc().append_child_value("manufacturer", "BioSemi")
    channels = info.desc().append_child("channels")
    for c in electrode:
        channels.append_child("channel")\
            .append_child_value("name", c)\
            .append_child_value("unit", "microvolts")\
            .append_child_value("type", "EEG")

    # next make an outlet; we set the transmission chunk size to 32 samples and
    # the outgoing buffer size to 360 seconds (max.)
    outlet = StreamOutlet(info, 32, 360)

    d1 = np.append(np.zeros((128,)), np.ones((128,)))
    d2 = np.append(np.linspace(0, 10, 10), np.linspace(0, 10, 10))
    print("now sending data...")
    ts, i = 0.001, 11
    while True:
        i = 11 if i == len(rows) else i
        sample = [float(x) for x in rows[i][2:2+8]]
        # sample[0] = d2[i % 20]
        # sample = [
        #     d1[i % 256], d2[i % 1000], d2[i % 1000], d1[i % 256],
        #     d2[i % 1000], d1[i % 256], d2[i % 1000], d1[i % 256]
        # ]
        outlet.push_sample(sample, timestamp=ts)
        i += 1
        ts += 0.001
        # get a time stamp in seconds (we pretend that our samples are actually
        # 125ms old, e.g., as if coming from some external hardware)
        # stamp = local_clock() - 0.001
        # now send it and wait for a bit

        
        # outlet.push_sample(sample)
        time.sleep(0.001)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        exit(0)
