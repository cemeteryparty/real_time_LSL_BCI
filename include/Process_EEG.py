""" Process EEG signal """
from scipy import signal
import numpy as np
import mne

""" Session EEG Signal by Slinding Window """
def EEGset_SW_Divider(EEGset, window_size=562, stride=281):
	n_chan, n_timep = EEGset.shape
	tstamps = []
	dataset = []
	for i in range(0, n_timep, stride):
		tmp = EEGset[:,i: i + window_size]
		if tmp.shape != (n_chan, window_size):
			break
		dataset.append(tmp)
		tstamps.append(i)
	dataset.append(EEGset[:, -window_size:])
	tstamps.append(EEGset.shape[1] - window_size)

	return np.array(dataset, dtype=np.float64), np.array(tstamps, dtype=np.int32)

""" reconstruct EEGset from Session """
def reconstruct_EEGset(Sess, TStamps, shape, window_size=562):
	EEGset = np.zeros(shape, dtype=np.float64)
	hcoef = np.zeros(shape[1], dtype=np.float64)
	
	hwin = signal.hann(window_size) + 1e-9
	for i in range(TStamps.size):
		tp = TStamps[i]
		EEGset[:, tp: tp+window_size] += Sess[i] * hwin
		hcoef[tp: tp+window_size] += hwin
	EEGset /= hcoef
	return EEGset
