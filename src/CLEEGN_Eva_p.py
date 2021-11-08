#!/opt/anaconda3/envs/pylab/bin/python3
import sys
if "include" not in sys.path:
	sys.path.insert(1, "include") # IMPORTANT

from DS_Gen import Dataset_Generator
from Process_EEG import EEGset_SW_Divider

from tensorflow.keras import utils
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from scipy.io import loadmat, savemat
import mne, os, time, math
import pandas as pd
import numpy as np

BASE_DIR = "/home/phlai410277/WorkSpace/RT_CLEEGN/"
BCI_202007 = "/mnt/left/phlai_DATA/bci-202007/"
BCI_CHALL_EVENT = ""
model_base_path = "/mnt/left/phlai_DATA/tmp/"
Window_Size = 512 # 2s
Stride = 256 # 1s:128
n_chan = 32

def isValidCMD():
	if len(sys.argv) != 3:
		return False
	elif sys.argv[1] not in ["1", "2", "3", "4"]:
		return False
	return True
def get_vid_list():
	return ["01", "02", "03"], ["04", "06"]
	fold_1 = ["01", "04", "07"]
	fold_2 = ["02", "05"]
	fold_3 = ["03", "06"]
	if sys.argv[1] == "1":
		return fold_2 + fold_3, fold_1
	if sys.argv[1] == "2":
		return fold_1 + fold_3, fold_2
	if sys.argv[1] == "3":
		return fold_1 + fold_2, fold_3
	return fold_2 + fold_3, fold_1

if not isValidCMD():
	print("usage: python3 CLEEGN_Eva_p.py vid ep")
	exit(2)

picks_tra, picks_tst = get_vid_list()
ep = sys.argv[2]; batch_size = 128

x_train, y_train = Dataset_Generator(BCI_202007 + "original/Data_S{sid}.set", BCI_202007 + "ica/Data_S{sid}.set", 
	n_chan, Window_Size, Stride, 128.0, 0.0, 10.0, picks_sid=picks_tra)
x_valid, y_valid = Dataset_Generator(BCI_202007 + "original/Data_S{sid}.set", BCI_202007 + "ica/Data_S{sid}.set", 
	n_chan, Window_Size, Stride, 128.0, 10.0, 15.0, picks_sid=picks_tra)
x_train = np.expand_dims(x_train, axis=3)
y_train = np.expand_dims(y_train, axis=3)
x_valid = np.expand_dims(x_valid, axis=3)
y_valid = np.expand_dims(y_valid, axis=3)
print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)

SimpleEva = 0
if SimpleEva:
	model = load_model(f"models/cv{sys.argv[1]}_tra.h5py")
	loss, _ = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
	val_loss, _ = model.evaluate(x_valid, y_valid, batch_size=batch_size, verbose=0)
	print("tra:", loss, val_loss)
	model = load_model(f"models/cv{sys.argv[1]}_val.h5py")
	loss, _ = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
	val_loss, _ = model.evaluate(x_valid, y_valid, batch_size=batch_size, verbose=0)
	print("val:", loss, val_loss)

	metrics = loadmat("output/metrics.mat")
	print(metrics["loss"].shape, metrics["val_loss"].shape)
	exit(0)

try:
	mat = loadmat("output/metrics.mat")
	logs = {"loss": mat["loss"].squeeze(), "val_loss": mat["val_loss"].squeeze()}
	min_loss = logs["loss"].min()
	min_vloss = logs["val_loss"].min()
except FileNotFoundError:
	logs = {"loss": [], "val_loss": []}
	min_loss = np.inf
	min_vloss = np.inf

nb_batch = math.ceil(x_train.shape[0] / batch_size)
tra_order = np.arange(x_train.shape[0])
np.random.shuffle(tra_order)
val_order = np.arange(x_valid.shape[0])
np.random.shuffle(val_order)
x_train = x_train[tra_order]; y_train = y_train[tra_order]
x_valid = x_valid[val_order]; y_valid = y_valid[val_order]
TimeStampStart = time.time()
for bs in range(nb_batch):
	mpath = model_base_path + f"cv{sys.argv[1]}_ep{ep}_{bs}.h5py"
	if os.path.exists(mpath):
		model = load_model(mpath)
		loss, _ = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
		vloss, _ = model.evaluate(x_valid, y_valid, batch_size=batch_size, verbose=0)

		print("\rEp{} {}/{} - {:.4f}s - loss = {:.8f} - val_loss = {:.8f}".format(ep, 
			bs + 1, nb_batch, time.time() - TimeStampStart, loss, vloss), end="")
		logs["loss"] = np.append(logs["loss"], loss)
		logs["val_loss"] = np.append(logs["val_loss"], vloss)
		if loss < min_loss:
			min_loss = loss
			model.save(f"models/cv{sys.argv[1]}_tra.h5py")
		if vloss < min_vloss:
			min_vloss = vloss
			model.save(f"models/cv{sys.argv[1]}_val.h5py")
	## FI
## DONE
print("\n")
savemat("output/metrics.mat", logs)
