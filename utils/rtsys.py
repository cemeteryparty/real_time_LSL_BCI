from pylsl import StreamOutlet
from pylsl import StreamInfo
import numpy as np
import time
import sys
import os


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
