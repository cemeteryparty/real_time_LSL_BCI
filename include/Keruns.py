""" Keras Custom training loop """
import sys, os, time, math
import numpy as np

from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Permute, BatchNormalization
from tensorflow.keras.models import Model

class SimpleKeTrainras(object):
	def __init__(self, model):
		super(SimpleKeTrainras, self).__init__()
		self.model = model
	def train(self, x_train, y_train, batch_size=128, epochs=20, shuffle=False, PREFIX=""):
		tra_order = np.arange(x_train.shape[0]); nb_batch = math.ceil(x_train.shape[0] / batch_size)
		for ep in range(epochs):
			print("\n\nEpoch {}/{}".format(ep + 1, epochs))
			if shuffle:
				np.random.shuffle(tra_order) # shuffle training order
			x_train_ = x_train[tra_order]; y_train_ = y_train[tra_order]
			TimeStampStart = time.time()
			for bs in range(nb_batch):
				self.model.train_on_batch(
					x_train_[bs*batch_size: (bs+1)*batch_size], y_train_[bs*batch_size: (bs+1)*batch_size]
				)

				self.model.save(f"/mnt/left/phlai_DATA/tmp/{PREFIX}_ep{ep}_{bs}.h5py")
				print("\r{}/{} - {:.4f}s".format(bs + 1, nb_batch, time.time() - TimeStampStart), end="")

class SimpleModelCkpt(object):
	def __init__(self, filepath, monitor, mode="min", verbose=1):
		super(SimpleModelCkpt, self).__init__()
		self.filepath = filepath
		self.monitor = monitor
		if mode == "min":
			self.mode = mode; self.bound = np.inf
		elif mode == "max":
			self.mode = mode; self.bound = -np.inf
		else:
			raise ValueError("mode in SimpleModelCkpt can only be \'min\' or \'max\'")
		self.verbose = verbose; self.lastlog = "Epoch ? bs ?: ???"
		self.metrics = np.zeros((3000, ), dtype=np.float32); self.__idx = 0
	def on_train_batch_end(self, epoch, batch, logs, model):
		if self.__idx >= self.metrics.size:
			self.metrics = np.append(self.metrics, np.zeros((1000, ), dtype=np.float32))
		self.metrics[self.__idx] = logs[self.monitor]; self.__idx += 1

		if ((self.mode == "min" and logs[self.monitor] < self.bound) or 
			(self.mode == "max" and logs[self.monitor] > self.bound)):

			self.lastlog = "Epoch {} bs{}: {} improved from {} to {}\n".format(
					epoch, batch, self.monitor, self.bound, logs[self.monitor]
				)
			if self.verbose == 1:
				print(self.lastlog)
			else:
				pass
			self.bound = logs[self.monitor]
			model.save(self.filepath)
		else:
			pass
	def on_train_end(self):
		self.metrics = self.metrics[:self.__idx]