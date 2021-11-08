""" Keras relative object """
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Lambda
from tensorflow.keras.layers import Dropout, Permute
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import backend as K
import numpy as np
import os, sys

""" CLEEGN: SCCNet based autoencoder """
def CLEEGN(Chans=64, Samples=128, N_F=20, fs=128.0):
	inputs = Input(shape=(Chans, Samples, 1))
	block1 = Conv2D(Chans, (Chans, 1), padding="valid", use_bias=True)(inputs)
	block1 = Permute((3, 2, 1))(block1)
	block1 = BatchNormalization()(block1)
	block1 = Conv2D(N_F, (1, int(fs * 0.1)), use_bias=True, padding="same")(block1)
	block1 = BatchNormalization()(block1)

	block2 = Conv2D(N_F, (1, int(fs * 0.1)), use_bias=True, padding="same")(block1)
	block2 = BatchNormalization()(block2)
	block2 = Conv2D(Chans, (Chans, 1), padding="same", use_bias=True)(block2)
	block2 = BatchNormalization()(block2)

	outputs = Conv2D(1, (Chans, 1), padding="same")(block2)
	return Model(inputs=inputs, outputs=outputs)

""" SCCNet """
def square(x):
	return K.square(x)
def SCCNet(nb_classes, Chans=64, Samples=128, fs=128, dropoutRate=0.5):
	# 62: 0.5s, 12: 0.1s
	inputs = Input(shape=(Chans, Samples, 1))
	block1 = Conv2D(Chans, (Chans, 1), padding="valid", use_bias=True)(inputs)
	block1 = Permute((3, 2, 1))(block1)
	block1 = BatchNormalization()(block1)
	
	block2 = Conv2D(20, (1, 12), use_bias=True, padding="same")(block1)
	block2 = BatchNormalization()(block2)
	block2 = Activation(square)(block2) # block2 = Lambda(lambda x: x ** 2)(block2)
	block2 = Dropout(rate=dropoutRate)(block2)

	block3 = AveragePooling2D(pool_size=(1, int(0.5 * fs)), strides=(1, int(0.1 * fs)))(block2) # default: padding="valid"
	
	block4 = Flatten()(block3)
	outputs = Dense(units=nb_classes, use_bias=True, activation="softmax")(block4)

	return Model(inputs=inputs, outputs=outputs)

""" new model ckpt """
class MyModelCkpt(keras.callbacks.Callback):
	def __init__(self, filepath, monitor="val_loss", mode="min", verbose=1):
		super().__init__()
		self.filepath = filepath
		self.monitor = monitor
		if mode == "min":
			self.mode = mode; self.bound = np.inf
		elif mode == "max":
			self.mode = mode; self.bound = -np.inf
		else:
			raise ValueError("mode in MyModelCkpt can only be \'min\' or \'max\'")
		self.verbose = verbose
		self.ckptlog = "?: No previous message"
	def on_epoch_end(self, epoch, logs=None):
		# print("[Debug]", self.mode, logs[self.monitor], self.bound)
		if ((self.mode == "min" and logs[self.monitor] < self.bound) or 
			(self.mode == "max" and logs[self.monitor] > self.bound)):
			# print("[Debug]", "best model")

			self.__dumpStr("Epoch {} : {} improved from {} to {}\n".format(
				epoch, self.monitor, self.bound, logs[self.monitor])
			)
			self.bound = logs[self.monitor]
			self.model.save(self.filepath)
		else:
			pass
	def __dumpStr(self, str_, end="\n"):
		self.ckptlog = str_
		if self.verbose == 1:
			print("\n{}".format(self.ckptlog), end=end)
		else:
			pass
