""" Dataset Generate Function """
from Process_EEG import EEGset_SW_Divider
import numpy as np
import mne

""" Split EEG by time """
def Split_EEG_by_Time(EEGset_x, EEGset_y, fs, tmin, tmax, ttype="sec"):
	offset = (np.array([tmin, tmax]) * fs).astype("int32")
	if ttype == "min":
		offset *= 60
	if offset[0] < 0:
		offset[0] = 0
	if offset[1] > EEGset_x.shape[1]:
		offset[1] = EEGset_x.shape[1]
	return EEGset_x[:, offset[0]:offset[1]+1], EEGset_y[:, offset[0]:offset[1]+1]

def Dataset_Generator(x_fpath, y_fpath, n_chan, win_size, stride, fs, tmin, tmax, picks_sid=[]):
	x, y = np.empty((0, n_chan, win_size)), np.empty((0, n_chan, win_size))
	for sid in picks_sid:
		ori_raw = mne.io.read_raw_eeglab(x_fpath.format(sid=sid))
		ica_raw = mne.io.read_raw_eeglab(y_fpath.format(sid=sid))
		ori_content = ori_raw.get_data() * 1e5
		ica_content = ica_raw.get_data() * 1e5

		x_, y_ = Split_EEG_by_Time(ori_content, ica_content, fs, tmin, tmax, ttype="min")
		del ori_content, ica_content # save ram
		x_, _ = EEGset_SW_Divider(x_, window_size=win_size, stride=stride)
		y_, _ = EEGset_SW_Divider(y_, window_size=win_size, stride=stride)

		x = np.vstack([x, x_])
		y = np.vstack([y, y_])
	return x, y