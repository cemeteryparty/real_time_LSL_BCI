from utils.rtsys import SigIntercept

from utils.cleegn import CLEEGN
from torchinfo import summary
import torch

import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.signal import oaconvolve
from scipy.signal import minimum_phase
from scipy.signal import firwin
import numpy as np
import math
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_length_factors = dict(hann=3.1, hamming=0.5, blackman=5.0) # hamming=3.3

def _create_filter(l_freq, h_freq, fs, l_trans_bw=1, h_trans_bw=1, window="hamming", phase="causal"):
    # https://github.com/mne-tools/mne-python/blob/maint/1.5/mne/filter.py#L1126
    # https://github.com/mne-tools/mne-python/blob/maint/1.5/mne/filter.py#L473

    """ filter_order """
    N = _length_factors["hamming"] / (1 / fs)
    N = math.ceil(N / 2) * 2 + 1
    if phase == "causal":
        N = N * 2 - 1

    """ construct transition gain """
    freq = [l_freq - l_trans_bw, l_freq, h_freq, h_freq + h_trans_bw]
    gain = [0, 1, 1, 0]
    if freq[-1] != fs / 2.0:
        freq += [fs / 2.0]
        gain += [0]
    if freq[0] != 0:
        freq = [0] + freq
        gain = [0] + gain
    freq = np.array(freq) / (fs / 2.0)

    """ construct filter response """
    h = np.zeros(N, dtype=np.float32)
    prev_freq, prev_gain = freq[-1], gain[-1]
    for this_freq, this_gain in zip(freq[::-1][1:], gain[::-1][1:]):
        if this_gain != prev_gain:
            transition = (prev_freq - this_freq) / 2.0
            this_N = int(round(_length_factors[window] / transition))
            this_N += 1 - this_N % 2  # make it odd

            this_h = firwin(
                this_N,
                (prev_freq + this_freq) / 2.0,
                window=window,
                pass_zero=True,
                fs=freq[-1] * 2,
            )
            offset = (N - this_N) // 2
            if this_gain == 0:
                h[offset : N - offset] -= this_h
            else:
                h[offset : N - offset] += this_h
        prev_gain = this_gain
        prev_freq = this_freq
    if phase == "causal":
        h = minimum_phase(h)
    return h

class PreProcessing(SigIntercept):

    def __init__(self, l_freq, h_freq, sfreq_IS, sfreq_OS, parent):
        duration = 30.0
        n_chan = parent.bufShape[0]
        super(PreProcessing, self).__init__(
            (n_chan, math.ceil(duration * sfreq_OS)),
            raiseStream=False, name="hidden_1", sfreq=sfreq_OS, parent=parent
        )

        self.sfreq_IS, self.sfreq_OS = sfreq_IS, sfreq_OS
        self.filt = _create_filter(l_freq, h_freq, fs=self.sfreq_IS)
        #self.b_notch, self.a_notch = signal.iirnotch(w0=50.0, Q=20.0, fs=fsIn)  #Lab60, BCIERN50
        self.x_part = np.zeros((self.bufShape[0], 0), dtype=np.float32)

    def step(self):
        filt = self.filt
        buffer = self.parent.buffer
        n_chan = buffer.shape[0]
        offset_beg = np.searchsorted(self.parent.flags, self.nextFID, side="left")
        xLen = buffer.shape[1] - offset_beg

        self.x_part = np.append(self.x_part, buffer[:, -xLen:], axis=1)

        """ band pass filtering """
        xf_part = np.zeros(
            (n_chan, self.x_part.shape[1] - len(filt) + 1), dtype=np.float32)
        for c in range(n_chan):
            xf_part[c] = oaconvolve(self.x_part[c], filt, mode="valid")
        # zero-padding: edge effect, leave out of valid to next iteration
        self.x_part = self.x_part[:, -(len(filt) - 1):]

        """ average rereference """
        ##xf_part -= xf_part.mean(axis=0)

        """ resample """
        t_bef = np.linspace(0, xf_part.shape[1] / self.sfreq_IS, xf_part.shape[1], endpoint=False)
        t_aft = np.linspace(
            0, xf_part.shape[1] / self.sfreq_IS,
            int(xf_part.shape[1] / self.sfreq_IS * self.sfreq_OS), endpoint=False
        )
        x_new = np.zeros((n_chan, t_aft.size))
        for c in range(n_chan):
            x_new[c] = np.interp(t_aft, t_bef, xf_part[c])
        return x_new

class ChannelNorm(SigIntercept):

    def __init__(self, sfreq_OS, parent):
        duration = 30.0
        n_chan = parent.bufShape[0]
        super(ChannelNorm, self).__init__(
            (n_chan, math.ceil(duration * sfreq_OS)),
            raiseStream=True, name="Pre-pocessedEEG", sfreq=sfreq_OS, parent=parent
        )

    def update(self, alpha=0.6):  # overridding `SigIntercept` update()
        # sync, add feature to SigIntercept?
        n_chan = self.buffer.shape[0]
        self.tstmps = self.parent.tstmps  # point to parent tstmps
        self.flags = self.parent.flags  # point to parent flags

        self.buffer = self.parent.buffer.copy()
        offset_beg = np.searchsorted(self.flags, 0, side="left")
        activated_buf = self.buffer[:, offset_beg:].copy()
        for c in range(n_chan):
            # # adaptive z-norm
            # s = activated_buf[c].std() * alpha
            # s = activated_buf[c,
            #     (-s < activated_buf[c]) & (activated_buf[c] < s)].std()
            # self.buffer[c, offset_beg:] /= s

            # mu, s = activated_buf[c].mean(), activated_buf[c].std() * alpha
            # cent_dt = activated_buf[c,
            #     (mu - s < activated_buf[c]) & (activated_buf[c] < mu + s)]
            # mu, s = cent_dt.mean(), cent_dt.std()
            # self.buffer[c, offset_beg:] = (self.buffer[c, offset_beg:] - mu) / s

            sign = activated_buf[c, :].copy()
            lower_bound = np.percentile(sign, 30)
            upper_bound = np.percentile(sign, 70)
            sign_b = sign[(lower_bound < sign) & (sign < upper_bound)]
            mu, s = np.median(sign_b), np.std(sign_b)
            self.buffer[c, offset_beg:] = (self.buffer[c, offset_beg:] - mu) / s

        """ average rereference """
        self.buffer -= self.buffer.mean(axis=0)
        ## END_OF_FUNCTIONS ##
        self.nextFID += 1

class CLEEGNing(SigIntercept):

    def __init__(self, model_path, sfreq_OS, parent):
        duration = 30.0
        n_chan = parent.bufShape[0]
        super(CLEEGNing, self).__init__(
            (n_chan, math.ceil(duration * sfreq_OS)),
            raiseStream=True, name="CLEEGNedEEG", sfreq=sfreq_OS, parent=parent
        )
        state = torch.load(model_path, map_location="cpu")
        self.model = model = CLEEGN(**state["struct_args"]).to(device)
        self.model.load_state_dict(state["state_dict"])
        # summary(model, input_size=(64, 1, 20, 512))

    def step(self):
        n_chan = self.buffer.shape[0]
        delayFID = self.nextFID - (1 if self.nextFID else 0)
        off_0 = np.searchsorted(self.parent.flags, delayFID, side="left")
        off_1 = np.searchsorted(self.parent.flags, self.nextFID, side="left")

        abuf = self.parent.buffer # ascent (parent) buffer

        self.model.eval()
        x0 = abuf[:, off_0:]
        with torch.no_grad():
            x = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float)
            x = self.model(x).detach().cpu().squeeze().numpy()
        x_part0, x_part1 = x[:, :off_1 - off_0], x[:, off_1 - off_0:]
        # print(x_part0.shape, x_part1.shape, off_0, off_1)

        if off_0 < off_1: # edge effect supress: experimental
            effLen = off_1 - ((off_0 + off_1) >> 1)
            self.buffer[:, -effLen:] = x_part0[:, -effLen:]
        return x_part1
