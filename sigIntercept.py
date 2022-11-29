from pylsl import StreamInlet
from pylsl import StreamOutlet
from pylsl import StreamInfo

from multiprocessing import current_process
from multiprocessing import cpu_count
from multiprocessing import Process

from scipy import signal
import prettytable as pt
import numpy as np
import pylsl
import math
import time
import os


class StreamInfo_(StreamInfo):

    def __init__(self, stream):
        super(StreamInfo_, self).__init__()
        self.__streamInstance = stream

    def lsl_info(self):
        stream = self.lsl_stream()
        return [
            "StreamInfo",
            stream.name(),
            stream.type(),
            stream.channel_count(),
            stream.nominal_srate(),
            stream.source_id(),
        ]

    def lsl_stream(self):
        return self.__streamInstance

    def __str__(self):
        info = self.lsl_info()
        # return stream.as_xml()
        return "\n".join([
            " > Stream Name  : {}".format(info[1]),
            " > Stream Type  : {}".format(info[2]),
            " > # of Channel : {}".format(info[3]),
            " > Sampling Rate: {}".format(info[4]),
            " > Stream SrcID : {}".format(info[5])
        ])


class StreamerLSL():

    def __init__(self):
        pass


if __name__ == "__main__":
    print("looking for an EEG stream...")
    streamList = pylsl.resolve_stream("type", "EEG")
    streamList = [StreamInfo_(s) for s in streamList]

    tb = pt.PrettyTable()
    tb.field_names = ["sid", "name", "type", "#_channel", "srate", "srcID"]
    for sid, stream in enumerate(streamList):
        sinfo = stream.lsl_info()
        sinfo[0] = sid
        tb.add_row(sinfo)
        # print(stream)
    print(tb)
    streamID = input("Select steam... ")

    # while True:
    #     pull_kwargs = {"timeout": 1, "max_samples":32}
    #     chunk, timestamps = inlet.pull_chunk(**pull_kwargs)
    #     if not len(timestamps):
    #         break  # stream loss
    #     print(len(chunk), len(timestamps))
    # stream = StreamerLSL()
    # sample, timestamp = inlet.pull_chunk()
    # stream.initialize()
    # while True:
    #     stream.preprocess()
