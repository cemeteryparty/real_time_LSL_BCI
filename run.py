from pylsl import StreamInlet
from pylsl import StreamOutlet
from pylsl import StreamInfo

from multiprocessing import current_process
from multiprocessing import cpu_count
from multiprocessing import Process

from utils.cleegn import CLEEGN
from torch import from_numpy as np2TT
from torchinfo import summary
import torch

from scipy import signal
import prettytable as pt
import numpy as np
import pylsl
import math
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StreamInfo_():  # StreamInfo

    def __init__(self, stream):
        super(StreamInfo_, self).__init__()
        self.__streamInstance = stream
        self.name   = stream.name()
        self.type   = stream.type()
        self.srate  = stream.nominal_srate()
        self.n_chan = stream.channel_count()
        self.ssid   = stream.source_id()

    def lsl_stream(self):
        return self.__streamInstance

    def __str__(self):
        # return stream.as_xml()
        return "\n".join([
            " > Stream Name  : {}".format(self.name),
            " > Stream Type  : {}".format(self.type),
            " > # of Channel : {}".format(self.n_chan),
            " > Sampling Rate: {}".format(self.srate),
            " > Stream SrcID : {}".format(self.ssid)
        ])

class PreProc_Block(): # 

    def __init__(self, lowcut, highcut, fsIn, fsOut, chunkShape):
        self.n_channel, self.n_sample = chunkShape
        self.sos = self.__butter_bandpass(lowcut, highcut, fsIn, order=5)
        self.zi = signal.sosfilt_zi(self.sos)
        self.zi = np.repeat(np.expand_dims(self.zi, axis=1), self.n_channel, axis=1)
        #self.b, self.a = self.__butter_bandpass(lowcut, highcut, fsIn, order=5)
        #self.zi = signal.lfilter_zi(self.b, self.a)
        #self.zi = np.tile(self.zi, (self.n_channel, 1))
        self.step = math.ceil(fsIn / fsOut)
        self.indices = np.arange(0, self.n_sample, dtype=int)[::self.step]

    def __butter_bandpass(self, lowcut, highcut, fs, order=3):
        # nyq = 0.5 * fs
        # low, high = lowcut / nyq, highcut / nyq
        # b, a = signal.butter(order, [low, high], btype='band')
        # b, a = signal.butter(order, [lowcut, highcut], fs=fs, btype='band')
        #return b, a
        sos = signal.butter(order, [lowcut, highcut], fs=fs, btype='band', output='sos')
        return sos

    def process(self, chunk, timestamps):
        # Shape vertify, no exception handler currently (note: 0-padding, interpolation)
        if chunk.shape[0] != self.n_channel or chunk.shape[1] != self.n_sample:
            print("E: unmatched chunk size, no exception handler")
            exit(1)
        if timestamps.shape[0] != self.n_sample:
            print("E: unmatched timestamps size, no exception handler")
            exit(1)

        """ Common Average Re-reference (CAR) """
        chunk -= chunk.mean(axis=0)

        """ Band-pass filtering """
        chunk, self.zi = signal.sosfilt(self.sos, chunk, zi=self.zi)
        #chunk, self.zi = signal.lfilter(self.b, self.a, chunk, zi=self.zi)

        """ Downsample """
        indices = [idx for idx in self.indices if idx < self.n_sample]
        self.indices %= self.n_sample
        timestamps = timestamps[indices]
        chunk = chunk[:, indices]
        self.indices += self.step

        return chunk, timestamps


class ArtifactRm_Block():

    n_channel, win_size = 8, 512

    def __init__(self, model_path):
        # TODO: auto get n_channel, win_size from load_model
        self.buffer = np.zeros((self.n_channel, self.win_size), dtype=np.float32)
        state = torch.load(model_path, map_location="cpu")
        self.model = CLEEGN(n_chan=8, fs=128.0, N_F=8).to(device)
        self.model.load_state_dict(state["state_dict"])
        # summary(model, input_size=(64, 1, 8, 512))

    def process(self, chunk):
        self.buffer[:, :self.win_size - chunk.shape[1]] = self.buffer[:, chunk.shape[1]:]
        self.buffer[:, -chunk.shape[1]:] = chunk

        x = self.buffer
        x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)
        x = np2TT(x).to(device, dtype=torch.float)
        self.model.eval()
        x = self.model(x)
        x = x.view(self.n_channel, -1).detach().cpu().numpy()

        return x[:, -64: -32]


def main():
    print("looking for an EEG stream...")
    streamList = pylsl.resolve_stream("type", "EEG")
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
    inlet = StreamInlet(selcStream.lsl_stream())

    block1 = PreProc_Block(1, 40, selcStream.srate, 128.0, (8, 256))
    block2 = ArtifactRm_Block("torch_CLEEGN/tmpfile/bc-12_0010.12_3040.4_8ch/set_{}/{}.pth".format(1, "bc-8chan"))

    outlet_1_info = StreamInfo("Pre-pocessedEEG", 'EEG', selcStream.n_chan, 128.0, 'float32', "run0001");
    outlet_2_info = StreamInfo("CLEEGNedEEG", 'EEG', selcStream.n_chan, 128.0,'float32', "run0002");
    outlet_1 = StreamOutlet(outlet_1_info)
    outlet_2 = StreamOutlet(outlet_2_info)
    while True:
        pull_kwargs = {"timeout": 1, "max_samples": 256} # "max_samples": 256
        chunk, timestamps = inlet.pull_chunk(**pull_kwargs)
        chunk = np.asarray(chunk, dtype=np.float32).T
        timestamps = np.asarray(timestamps, dtype=np.float32)
        if not len(timestamps):
            print(f"[x] Loss conection to the stream: {selcStream.name()}...")
            break # TODO: try recovery???
        #print("read chunk", chunk.shape, timestamps.shape)
        chunk_1, timestamps = block1.process(chunk, timestamps)
        chunk_2 = block2.process(chunk_1)
        # print(chunk_1.shape, chunk_2.shape)

        #print(chunk.min(axis=1), chunk.max(axis=1))
        #chunk_1 = chunk_1.T
        outlet_1.push_chunk(chunk_1.T.tolist(), timestamp=timestamps[-1])
        outlet_2.push_chunk(chunk_2.T.tolist(), timestamp=timestamps[-1])
        #outlet_1.push_chunk(chunk.tolist())

    #     
    #     print(len(chunk), len(timestamps))
    # stream = StreamerLSL()
    # sample, timestamp = inlet.pull_chunk()
    # stream.initialize()
    # while True:
    #     stream.preprocess()

if __name__ == "__main__":
    # try:
    #     main()
    # except KeyboardInterrupt:
    #     print()
    #     exit(0)
    main()
