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

class SigIntercept(object):

    def __init__(
        self, bufferDim, bufferSize=1024, raiseStream=False, name=None, sfreq=None, parent=None
        ):

        self.bufShape = (bufferDim, bufferSize)
        self.buffer = np.zeros((bufferDim, bufferSize), dtype=np.float32)
        self.tstmps = np.zeros((bufferSize, ), dtype=np.float32) - 1
        self.flags = np.zeros((bufferSize, ), dtype=int) - 1

        if raiseStream:
            outletInfo = StreamInfo(name, "EEG", bufferDim, sfreq, "float32", f"run_{int(time.time())}")
            self.outlet = StreamOutlet(outletInfo)
        self.parent = parent
        self.latestDataLen = 0  # latest chunk start pos
        self.nextFID = 0  # next update flag id

    def update(self, x=None, t=None):
        if self.parent is not None:
            x, t = self.befed()
            # self.buffer = self.parent.buffer.copy()

        if (x is not None) and (t is not None):
            #xLen = x.shape[1]
            xLen = self.latestDataLen = x.shape[1]

            self.buffer[:, :self.bufShape[1] - xLen] = self.buffer[:, xLen:]
            self.buffer[:, -xLen:] = x

            self.tstmps[:self.bufShape[1] - xLen] = self.tstmps[xLen:]
            self.tstmps[-xLen:] = t

            self.flags[:self.bufShape[1] - xLen] = self.flags[xLen:]
            self.flags[-xLen:] = self.nextFID

            #self.latestDataLen = xLen
        else:
            raise ValueError("no valid data stream is avaliable")
        self.nextFID += 1

    def befed(self):
        pass  # update from parent buffer

    def send(self, delay=0):
        """
        Update stream
        Args:
            delay (int): number of delay chunk
        """
        useFlag = self.flags[-1] - delay
        if useFlag >= 0:
            offset_l = np.searchsorted(self.flags, useFlag, side="left")
            offset_r = np.searchsorted(self.flags, useFlag, side="right")
            x = self.buffer[:, offset_l: offset_r].copy()
            t = self.tstmps[offset_r - 1]
            self.outlet.push_chunk(x.T.tolist(), timestamp=t)
            return 0  # 
        return 1


class BasicRecv(SigIntercept):

    def __init__(self, n_channel, sfreq, bufferSize=None):
        if bufferSize is not None:
            super(BasicRecv, self).__init__(n_channel, bufferSize=bufferSize, sfreq=sfreq)
        else:
            super(BasicRecv, self).__init__(n_channel, sfreq=sfreq)


class PreProcessing(SigIntercept):

    def __init__(self, lowcut, highcut, fsIn, fsOut, parent):
        super(PreProcessing, self).__init__(
            parent.bufShape[0], raiseStream=True, name="Pre-pocessedEEG", sfreq=fsOut,
            parent=parent
        )
        
        self.sos = self.__butter_bandpass(lowcut, highcut, fsIn, order=5)
        self.zi = signal.sosfilt_zi(self.sos)
        self.zi = np.repeat(np.expand_dims(self.zi, axis=1), self.bufShape[0], axis=1)
        self.step = math.ceil(fsIn / fsOut)
        self.a0 = 0  # next start index, downsample issue

    def __butter_bandpass(self, lowcut, highcut, fs, order=3):
        sos = signal.butter(order, [lowcut, highcut], fs=fs, btype='band', output='sos')
        return sos

    def befed(self):
        xLen = self.parent.latestDataLen
        x = self.parent.buffer[:, -xLen:].copy()
        t = self.parent.tstmps[-xLen:].copy()

        """ Common Average Re-reference (CAR) """
        x -= x.mean(axis=0)

        """ Band-pass filtering """
        x, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)

        # """ Downsample """
        indices = np.arange(self.a0, x.shape[1], dtype=int)[::self.step]
        x, t = x[:, indices], t[indices]

        self.a0 = indices[-1] + self.step - xLen
        return x, t



class CLEEGNing(SigIntercept):

    def __init__(self, model_path, fsOut, parent):
        super(CLEEGNing, self).__init__(
            parent.bufShape[0], raiseStream=True, name="CLEEGNedEEG", sfreq=fsOut,
            parent=parent
        )
        # Done: auto get n_channel, win_size from load_model, next update
        state = torch.load(model_path, map_location="cpu")
        self.model = CLEEGN(n_chan=8, fs=128.0, N_F=8).to(device)
        self.model.load_state_dict(state["state_dict"])
        # summary(model, input_size=(64, 1, 8, 512))

        self.tstmps = self.parent.tstmps  # point to parent tstmps
        self.flags = self.parent.flags  # point to parent flags

    def update(self):
        self.model.eval()

        xLen, n_channel = 512, self.bufShape[0]
        x = self.parent.buffer[:, -xLen:].copy()

        x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)
        x = np2TT(x).to(device, dtype=torch.float)
        x = self.model(x)
        x = x.view(n_channel, -1).detach().cpu().numpy()
        self.buffer[:, -xLen:] = x

        #print(self.tstmps[-1], self.flags[-1])
        # return np.zeros((8, 0)), 0  # special design, skip update



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


    root = BasicRecv(8, selcStream.srate)
    """ temporary use local var, module it in future """
    block1 = PreProcessing(1, 40, selcStream.srate, 128.0, parent=root)
    block2 = CLEEGNing("torch_CLEEGN/tmpfile/bc-12_0010.12_3040.4_8ch/set_{}/{}.pth".format(1, "bc-8chan"), fsOut=128.0, parent=block1)

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
        block2.update()

        block1.send(delay=2)
        block2.send(delay=2)


if __name__ == "__main__":
    # try:
    #     main()
    # except KeyboardInterrupt:
    #     print()
    #     exit(0)
    main()
