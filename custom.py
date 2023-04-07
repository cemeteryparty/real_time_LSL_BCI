from utils.rtsys import SigIntercept

from utils.cleegn import CLEEGN
from torch import from_numpy as np2TT
from torchinfo import summary
import torch

from scipy import signal
import numpy as np
import math
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PreProcessing(SigIntercept):

    def __init__(self, lowcut, highcut, fsIn, fsOut, parent):
        # super(PreProcessing, self).__init__(
        #     parent.bufShape[0], raiseStream=True, name="Pre-pocessedEEG", sfreq=fsOut,
        #     parent=parent
        # )
        super(PreProcessing, self).__init__(
            parent.bufShape[0], raiseStream=False, name="xxx", sfreq=fsOut,
            parent=parent
        )
        
        self.sos = self.__butter_bandpass(lowcut, highcut, fsIn, order=3)
        self.zi = signal.sosfilt_zi(self.sos)
        self.zi = np.repeat(np.expand_dims(self.zi, axis=1), self.bufShape[0], axis=1)
        self.b_notch, self.a_notch = signal.iirnotch(w0=50.0, Q=20.0, fs=fsIn)  #Lab60, BCIERN50
        
        self.T = 1.0 / fsOut # out period

    def __butter_bandpass(self, lowcut, highcut, fs, order=3):
        sos = signal.butter(order, [lowcut, highcut], fs=fs, btype='band', output='sos')
        return sos

    def befed(self):
        buf = self.parent.buffer
        offset_l = np.searchsorted(self.parent.flags, self.nextFID, side="left")
        xLen = buf.shape[1] - offset_l

        x = buf[:, -xLen:].copy()
        t = self.parent.tstmps[-xLen:].copy()

        x -= x.mean(axis=1, keepdims=True) # channel 0-mean

        """ Norm to near 1 """
        for c in range(self.bufShape[0]):
            mu, s = buf[c, :].mean(), buf[c, :].std() * 2.0
            bar = (mu - s <= buf[c, :]) & (buf[c, :] <= mu + s)
            x[c, :] /= buf[c, bar].std()

        x -= x.mean(axis=0)  # avg_reref

        x = signal.filtfilt(self.b_notch, self.a_notch, x) # notch 60

        x, self.zi = signal.sosfilt(self.sos, x, zi=self.zi) # bpf 1,40

        """ Downsample """
        t_new = np.arange(t[0], t[-1] + self.T, self.T)
        x_new = np.zeros((8, t_new.size))
        for c in range(self.bufShape[0]):
            x_new[c] = np.interp(t_new, t, x[c])
        return x_new, t_new

class ChannelShift(SigIntercept):

    def __init__(self, fsOut, parent):
        super(ChannelShift, self).__init__(
            parent.bufShape[0], raiseStream=True, name="Pre-pocessedEEG", sfreq=fsOut,
            parent=parent
        )
        # sigmas = [
        #     5.353120,14.690568,7.081487,6.351827,6.620898,9.770247,6.404835,8.365208
        # ]
        # self.sigmas = np.asarray(sigmas, dtype=np.float32)
        self.sigmas = np.ones((self.bufShape[0], ), dtype=np.float32) * 7

        # sync, add feature to SigIntercept?
        self.tstmps = self.parent.tstmps  # point to parent tstmps
        self.flags = self.parent.flags  # point to parent flags

    def update(self):  # overridding `SigIntercept` update()
        stds = self.sigmas
        self.buffer = self.parent.buffer.copy()
        coef = self.buffer.std(axis=1) / stds
        coef = np.expand_dims(coef, axis=-1)
        self.buffer /= coef

        # self.buffer *= 1e6  # BCI_ERN, easy
        # self.buffer = self.parent.buffer.copy()  # no shift

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

    def update(self):  # overridding `SigIntercept` update()
        self.buffer = self.parent.buffer.copy()
        self.model.eval()

        xLen, n_channel = 512, self.bufShape[0]
        x = self.buffer[:, -xLen:]

        x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)
        x = np2TT(x).to(device, dtype=torch.float)
        x = self.model(x)
        x = x.view(n_channel, -1).detach().cpu().numpy()
        self.buffer[:, -xLen:] = x
