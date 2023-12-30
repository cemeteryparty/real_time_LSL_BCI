from pylsl import StreamOutlet
from pylsl import StreamInfo
from scipy.io import savemat
import numpy as np
import math
import time
import sys
import os


class SigIntercept(object):

    def __init__(
        self, bufShape, raiseStream=False, name=None, sfreq=None, parent=None
        ):

        assert len(bufShape) == 2
        n_chan, n_timestamps = bufShape
        self.bufShape = (n_chan, n_timestamps)
        self.buffer = np.zeros(bufShape, dtype=np.float32)
        #self.tstmps = np.zeros((bufferSize, ), dtype=np.float32) - 1
        self.tstmps = np.arange(n_timestamps) / sfreq
        self.flags = np.zeros((n_timestamps, ), dtype=int) - 1

        if raiseStream:
            outletInfo = StreamInfo(name, "EEG", n_chan, sfreq, "float32", f"run_{int(time.time())}")
            self.outlet = StreamOutlet(outletInfo)
        self.parent = parent
        self.nextFID = 0  # next update flag id
        self.sfreq = sfreq

    def update(self, x):
        if x is None:
            raise ValueError("stop it, get some help")

        xLen = x.shape[1]
        self.buffer[:, :self.bufShape[1] - xLen] = self.buffer[:, xLen:]
        self.buffer[:, -xLen:] = x

        self.tstmps += (x.shape[1] / self.sfreq)
        self.flags[:self.bufShape[1] - xLen] = self.flags[xLen:]
        self.flags[-xLen:] = self.nextFID

        self.nextFID += 1

    def step(self):
        pass  # update from parent buffer

    def send(self, delay=0, picks=None):
        """
        TODO: try to make a multi-process
        Update stream
        Args:
            delay (int): number of delay chunk
        """
        picks = np.arange(self.bufShape[0]) if picks is None else picks

        useFlag = self.flags[-1] - delay
        if useFlag >= 0:
            off_0 = np.searchsorted(self.flags, useFlag, side="left")
            off_1 = np.searchsorted(self.flags, useFlag, side="right")
            x = self.buffer[picks, off_0: off_1].copy()
            t = self.tstmps[off_1 - 1]
            self.outlet.push_chunk(x.T.tolist(), timestamp=t)
            return 0  # 
        return 1

class BasicRecv(SigIntercept):

    def __init__(self, n_channel, sfreq, bufferSize=None):
        n_timestamps = bufferSize if bufferSize is not None else math.ceil(30.0 * sfreq)
        super(BasicRecv, self).__init__((n_channel, n_timestamps), sfreq=sfreq)
