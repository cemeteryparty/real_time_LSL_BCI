from utils.decos import StreamInfo_
from utils.rtsys import SigIntercept
from utils.rtsys import BasicRecv

from custom import PreProcessing
from custom import ChannelShift
from custom import CLEEGNing

import prettytable as pt
import numpy as np
import pylsl
import math
import time
import os


def main():
    print("looking for an EEG stream...")
    streamList = pylsl.resolve_streams()
    streamList = [StreamInfo_(s) for s in streamList]

    tb = pt.PrettyTable()
    tb.field_names = ["sid", "name", "type", "#_channel", "srate", "srcID"]
    for sid, stm in enumerate(streamList):
        sinfo = [sid, stm.name, stm.type, stm.n_chan, stm.srate, stm.ssid]
        tb.add_row(sinfo)
        # print(stm)
    print(tb)
    streamID = int(input("Select steam... "))
    selcStream = streamList[streamID]
    inlet = pylsl.StreamInlet(selcStream.lsl_stream())


    root = BasicRecv(8, selcStream.srate)
    block1 = PreProcessing(1, 40, selcStream.srate, 128.0, parent=root)
    block1_cs = ChannelShift(fsOut=128.0, parent=block1)
    block2 = CLEEGNing(
        "torch_CLEEGN/tmpfile/bc-12_0010.12_3040.4_8ch/set_{}/{}.pth".format(1, "bc-8chan"),
        fsOut=128.0, parent=block1_cs
    )

    while True:
        pull_kwargs = {"timeout": 1, "max_samples": 256}
        chunk, timestamps = inlet.pull_chunk(**pull_kwargs)
        chunk = np.asarray(chunk, dtype=np.float32).T
        timestamps = np.asarray(timestamps, dtype=np.float32)
        if not len(timestamps):
            print(f"[x] Loss conection to the stream: {selcStream.name()}...")
            break # TODO: try recovery???

        root.update(chunk, timestamps)
        block1.update()
        block1_cs.update()
        block2.update()

        block1_cs.send(delay=2)
        block2.send(delay=2)


if __name__ == "__main__":
    # try:
    #     main()
    # except KeyboardInterrupt:
    #     print()
    #     exit(0)
    main()
