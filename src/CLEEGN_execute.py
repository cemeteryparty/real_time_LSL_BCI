#!/opt/anaconda3/envs/pylab/bin/python3
import sys
if "include" not in sys.path:
	sys.path.insert(1, "include") # IMPORTANT

from Process_EEG import EEGset_SW_Divider, reconstruct_EEGset
from KerasObj import CLEEGN, SCCNet

from tensorflow.keras import utils
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from scipy.io import loadmat, savemat
from scipy import signal
import mne, os, time
import pandas as pd
import numpy as np

BDIR = "/content/drive/MyDrive/CLEEGN/"
BCI_CHALL_DIR = "/mnt/left/phlai_DATA/bci-challenge/"
BCI_CHALL_EVENT = "/mnt/left/phlai_DATA/bci-challenge-event/"
Window_Size = 512
Stride = 64 # 1s:128
n_chan = 56

# m_path = "EndBata_0-10t30-40v_W4s_ST2s_bs128_1e-3"; c_type = "val"
m_path = sys.argv[1]; c_type = sys.argv[2]
TIME_Start = time.time()
sbjs = ['02', '06', '07', '11', '12', '13', '14', '16', '17', '18', '20', '21', '22', '23', '24', '26']
mse_mat = np.zeros((16, 3), dtype=np.float32)
for si in range(16):
	sid = sbjs[si]
	event_mat = loadmat(BCI_CHALL_EVENT + f"Data_S{sid}_event.mat")
	ori_raw = mne.io.read_raw_eeglab(BCI_CHALL_DIR + f"original/Data_S{sid}.set")
	ori_content_1e6 = ori_raw.get_data() * 1e6
	mne.rename_channels(ori_raw.info, mapping = {"P08": "PO8"})
	electrode = ori_raw.ch_names
	montage_1020 = mne.channels.make_standard_montage("standard_1020")

	position = montage_1020.get_positions()["ch_pos"]
	position = np.asarray([position[i] for i in electrode])

	vid = ["06", "13", "18", "14", 
		"22", "26", "24", "02", 
		"21", "12", "20", "23", 
		"07", "16", "17", "11"
	].index(sid) // 4 + 1

	model_path = f"models/{m_path}/cv{vid}_{c_type}.h5py"
	model = load_model(model_path)
	print(f"S{sid} use {model_path}")

	x_train, tstmp = EEGset_SW_Divider(ori_content_1e6, window_size=Window_Size, stride=Stride)
	x_train = np.expand_dims(x_train, axis=3)

	y = model.predict(x_train, batch_size=128).squeeze()
	y_cleegn = reconstruct_EEGset(y, tstmp, ori_content_1e6.shape, window_size=Window_Size)

	mse = MeanSquaredError()
	ica_raw = mne.io.read_raw_eeglab(f"/mnt/left/lambert/ner2015_together/fulldata/onlybrain/Data_S{sid}_Sess.set", verbose=0)
	ica_content_1e6 = ica_raw.get_data() * 1e6
	mse_mat[si][0] = mse(ori_content_1e6, ica_content_1e6).numpy()
	mse_mat[si][1] = mse(ori_content_1e6, y_cleegn).numpy()
	mse_mat[si][2] = mse(y_cleegn, ica_content_1e6).numpy()

	continue # NOT dump mne file

	cleegn_raw = mne.io.RawArray(y_cleegn / 1e6, ori_raw.info)
	anno = mne.annotations_from_events(events=event_mat["events"], sfreq=cleegn_raw.info["sfreq"], 
		event_desc={0: "worse/feedback", 1: "good/feedback"}, orig_time=cleegn_raw.info['meas_date'])
	try:
		cleegn_raw.set_annotations(anno)
	except TypeError:
		pass
	# cleegn_raw.save(f"fulldata/Data_S{sid}_raw.fif", overwrite=True)
	cleegn_raw.save("fulldata/Data_Sxx_raw.fif", overwrite=True)

df = pd.DataFrame(mse_mat, columns=["MSE_OI", "MSE_OC", "MSE_IC"], index=sbjs)
df.to_csv(f"output/{m_path}_{c_type}.csv")

print(f"\n\nRuntime: {time.time() - TIME_Start} sec")