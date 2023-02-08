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
        
        self.sos = self.__butter_bandpass(lowcut, highcut, fsIn, order=5)
        self.zi = signal.sosfilt_zi(self.sos)
        self.zi = np.repeat(np.expand_dims(self.zi, axis=1), self.bufShape[0], axis=1)
        self.step = math.ceil(fsIn / fsOut)
        self.a0 = 0  # next start index, downsample issue
        # os.system("echo \"{}\" > saved.csv".format(
        #     "Fp1,Fp2,T7,T8,O1,O2,Fz,Pz"))  # reset

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
        # for ix in range(x.shape[1]):
        #     l = x[:, ix].tolist()
        #     os.system("echo \"{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\" >> saved.csv".format(
        #         *l))
        return x, t

class ChannelShift(SigIntercept):

    def __init__(self, fsOut, parent):
        super(ChannelShift, self).__init__(
            parent.bufShape[0], raiseStream=True, name="Pre-pocessedEEG", sfreq=fsOut,
            parent=parent
        )
        sigmas = [
            5.353120,14.690568,7.081487,6.351827,6.620898,9.770247,6.404835,8.365208
        ]  ######## to be updated
        self.sigmas = np.asarray(sigmas, dtype=np.float32)

        # sync, add feature to SigIntercept?
        self.tstmps = self.parent.tstmps  # point to parent tstmps
        self.flags = self.parent.flags  # point to parent flags

    def update(self):  # overridding `SigIntercept` update()
        stds = self.sigmas
        self.buffer = self.parent.buffer.copy()
        coef = self.buffer.std(axis=1) / stds
        coef = np.expand_dims(coef, axis=-1)
        self.buffer /= coef
        # self.buffer *= 1e6  # BCI_ERN

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
        self.model.eval()

        xLen, n_channel = 512, self.bufShape[0]
        x = self.parent.buffer[:, -xLen:].copy()

        x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)
        x = np2TT(x).to(device, dtype=torch.float)
        x = self.model(x)
        x = x.view(n_channel, -1).detach().cpu().numpy()
        self.buffer[:, -xLen:] = x
