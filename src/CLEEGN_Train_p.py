#!/opt/anaconda3/envs/pylab/bin/python3
import sys
if "include" not in sys.path:
	sys.path.insert(1, "include") # IMPORTANT

from DS_Gen import Dataset_Generator
from Process_EEG import EEGset_SW_Divider
from KerasObj import CLEEGN

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
Window_Size = 512 # 2s
Stride = 256 # 1s:128
n_chan = 32

def isValidCMD():
	if len(sys.argv) != 2:
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
	print("usage: ./CLEEGN_Train.py vid")
	exit(2)

picks_tra, picks_tst = get_vid_list()
# picks_tra, picks_tst = ["06", "13", "18", "14"], ["06", "13", "18", "14"]

x_train, y_train = Dataset_Generator(BCI_202007 + "original/Data_S{sid}.set", BCI_202007 + "ica/Data_S{sid}.set", 
	n_chan, Window_Size, Stride, 128.0, 0.0, 10.0, picks_sid=picks_tra)
x_valid, y_valid = Dataset_Generator(BCI_202007 + "original/Data_S{sid}.set", BCI_202007 + "ica/Data_S{sid}.set", 
	n_chan, Window_Size, Stride, 128.0, 10.0, 15.0, picks_sid=picks_tra)

x_train = np.expand_dims(x_train, axis=3)
y_train = np.expand_dims(y_train, axis=3)
x_valid = np.expand_dims(x_valid, axis=3)
y_valid = np.expand_dims(y_valid, axis=3)
print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)
#exit(0)
cleegn = CLEEGN(Chans=n_chan, Samples=Window_Size, N_F=56, fs=128.0)
cleegn.compile(loss = MeanSquaredError(), optimizer=Adam(learning_rate=1e-3), metrics=["acc"])

from Keruns import SimpleKeTrainras
kt = SimpleKeTrainras(cleegn)
kt.train(x_train, y_train, batch_size=128, epochs=30, shuffle=True, PREFIX=f"cv{sys.argv[1]}")
