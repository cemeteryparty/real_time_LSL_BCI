from Process_EEG import EEGset_SW_Divider, reconstruct_EEGset
from tensorflow.keras.models import load_model
from scipy.io import loadmat, savemat
import numpy as np
import os

""" Generate Epoch mat file """
def Epoch_mat_Gen(EEG_Data, events, fs=256.0, tmin=-2.0, tmax=5.0, filepath="./tmp.mat", model="none"):
	"""
	EEG_Data: (chan, timepoint), fs: sampling frequency
	events: (timestamp, error, event code)
	"""
	if len(np.array(EEG_Data).shape) != 2:
		raise ValueError("EEG_Data should in shape (chan, timepoint)")
	offset = (np.array([tmin, tmax]) * fs).astype("int32")
	epochs, labels = [], np.expand_dims(events[:, 2], axis=1)
	for event in events:
		epochs.append(EEG_Data[:, event[0]+offset[0]: event[0]+offset[1]])
	session_content = {"model": model, 
		"x_test": np.array(epochs), "y_test": labels}
	savemat(filepath, session_content)
	return session_content

""" Generate Epoch mat file from short timeline """
def ReEpoch_mat_Gen(EEG_Data, events, win_size, stride, fs, tmin, tmax, filepath="./tmp.mat", modelpath="none"):
	"""
	EEG_Data: (chan, timepoint), fs: sampling frequency
	events: (timestamp, error, event code)
	"""
	if len(np.array(EEG_Data).shape) != 2:
		raise ValueError("EEG_Data should in shape (chan, timepoint)")
	if not os.path.exists(modelpath):
		raise Exception(f"E: Model {modelpath} doesn't exist.")
	
	model = load_model(modelpath)
	offset = (np.array([tmin, tmax]) * fs).astype("int32")
	epochs, labels = [], np.expand_dims(events[:, 2], axis=1)
	for event in events:
		trange = (np.array([-2, 2]) * win_size).astype("int32") + event[0]
		trange[0] = 0 if trange[0] < 0 else trange[0]
		trange[1] = EEG_Data.shape[1] if trange[1] > EEG_Data.shape[1] else trange[1]
		x_content = EEG_Data[:, trange[0]: trange[1]]

		x_pred, tstamps = EEGset_SW_Divider(x_content, window_size=win_size, stride=stride)
		x_pred = np.expand_dims(x_pred, axis=3)
			
		y_pred = model.predict(x_pred)
		y_pred = np.reshape(y_pred, y_pred.shape[:3])

		re_x_content = reconstruct_EEGset(Epochs=y_pred, TStamps=tstamps, n_chan=56, 
			n_timepoint=x_content.shape[1], window_size=win_size, stride=stride)

		epochs.append(re_x_content[:, event[0] - trange[0] + offset[0]: event[0] - trange[0] + offset[1]])
	session_content = {"model": modelpath, "x_test": np.array(epochs), "y_test": labels}
	savemat(filepath, session_content)
	return session_content