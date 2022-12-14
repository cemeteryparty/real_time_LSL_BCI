from torch_CLEEGN.utils.cleegn import CLEEGN
from torch import from_numpy as np2TT
from torchinfo import summary
import torch

from pylsl import StreamInlet
from pylsl import StreamOutlet
from pylsl import StreamInfo
from pylsl import resolve_stream

from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt

# from tensorflow.keras.models import load_model
from scipy import signal
import numpy as np
import pylsl
import math
import time
import copy
# import mne
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
for stream in streams:
    print(stream.name())
""" model = load_model("bci_8chan.h5")
model.summary() """
state_path = os.path.join(
    "torch_CLEEGN/tmpfile/bc-12_0010.12_3040.4_8ch",
    "set_{}/{}.pth".format(1, "bc-8chan")
)
state = torch.load(state_path, map_location="cpu")
model = CLEEGN(n_chan=8, fs=128.0, N_F=8).to(device)
model.load_state_dict(state["state_dict"])
summary(model, input_size=(64, 1, 8, 512))

print("model loaded")
# # create a new inlet to read from the stream
inlet = StreamInlet(streams[0])


## class StreamerLSL(QThread):
class StreamerLSL():
    ## sendData = pyqtSignal(list, list, list)
    stream_params = {'chunk_idx': 0, 'metadata': {}, 'srate': 128, 'chunkSize': 256, 'forwardSize': 512,
                        'downSampling': None, 'downSamplingFactor': None, 'downSamplingBuffer1': None, 
                        'downSamplingBuffer2': None, 'inlet': None, 'stream_idx': None, 'is_marker': False}
    def __init__(self):
        super().__init__()
        self.eeg_channels = 8
        self.model_input_chan = 8
        self.output_sample_rate = 128
        self.send_data1 =[]
        self.send_data2 =[]
        self.send_ts =[]
        ## self.sendData.connect(self.update_stream)
        self.dataBuffer = np.zeros((0,self.eeg_channels))
        self.tsBuffer = np.array([])
        self.zi = None
        
        self.stream1_name = "prepocessed data"
        self.stream2_name = "after CLEEGN data"
        self.stream1_id = "1234"
        self.stream2_id = "1234"

        print ("Creating LSL stream for EEG. Name:", self.stream1_name, "- data type: float32.", self.eeg_channels, "channels at", self.output_sample_rate, "Hz.")
        self.info_stream1 = StreamInfo(self.stream1_name, 'EEG', self.model_input_chan, self.output_sample_rate,'float32',self.stream1_id);
        self.info_stream2 = StreamInfo(self.stream2_name, 'EEG', self.model_input_chan, self.output_sample_rate,'float32',self.stream2_id);

        # make outlets
        self.outlet_stream1 = StreamOutlet(self.info_stream1)
        self.outlet_stream2 = StreamOutlet(self.info_stream2)

        self.stream_params['inlet'] = pylsl.StreamInlet(streams[0])  # ?????
        srate = streams[0].nominal_srate()
        self.stream_params['downSampling'] = srate > self.output_sample_rate
        # stream_params['chunkSize'] = round(srate / self.chunksPerScreen * self.seconds_per_screen)
        if self.stream_params['downSampling']:
            self.stream_params['downSamplingFactor'] = round(srate / self.output_sample_rate)
            n_buff = round(self.stream_params['chunkSize'] / self.stream_params['downSamplingFactor'])
            self.stream_params['downSamplingBuffer1'] = [[0] * int(streams[0].channel_count()) for _ in range(n_buff)]
            self.stream_params['downSamplingBuffer2'] = [[0] * int(streams[0].channel_count()) for _ in range(n_buff)]
    
    def butter_bandpass(self, lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut,  highcut, fs, initial, order=10):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        zf = None
        if(initial is None):
            y= signal.lfilter(b, a, data)
        else:
            y, zf = signal.lfilter(b, a, data, zi = initial)
        return y, zf

    def initialize(self):
        b, a = self.butter_bandpass(1, 40, 1024, order=5) # 1024
        self.zi = signal.lfilter_zi(b, a)
        self.zi = np.tile(self.zi, (self.eeg_channels, 1))
        # print(self.zi)
        # print(b, a)
        # exit(0)

    def preprocess(self):
        
        self.send_ts, self.send_data1, self.send_data2 = [], [], [] 
        # params = copy.deepcopy(self.stream_params)
        inlet = self.stream_params['inlet']
        pull_kwargs = {'timeout': 1}
        if self.stream_params['chunkSize']:
            pull_kwargs['max_samples'] = self.stream_params['chunkSize']
        
        '''pull chunk from LSL'''
        sample, timestamp = inlet.pull_chunk(**pull_kwargs)
        self.send_data1 = copy.deepcopy(sample)
        self.send_data2 = copy.deepcopy(sample)
        self.send_ts = timestamp
        current_time =  time.time()
        sample = np.array(sample)
        timestamp = np.array(timestamp)
        print("read chunk shape:", sample.shape)

        ## removal of the mean value of the signal
        #self.send_data1 = self.send_data1 - np.mean(self.send_data1, axis=0)  # CAR
        self.send_data1 = self.send_data1 - np.expand_dims(np.mean(self.send_data1, axis=1), axis=-1)

        # print(self.send_data1.shape) # (256, 8)
        self.send_data1 = np.array(self.send_data1).T
        #print(self.send_data1.mean(axis=1)) # (8, 256)

        '''Band pass filtering'''
        order = 5
        hp_cutoff = 40
        lp_cutoff = 1
        nyq = 1000 / 2
        

        #x = self.butter_bandpass_filter(np.array(self.send_data1).T, lp_cutoff, hp_cutoff, 1000, order=5)
        x, self.zi = self.butter_bandpass_filter(self.send_data1, lp_cutoff, hp_cutoff, 1024, initial=self.zi, order=5)
        # print("max value: ", np.max(x))
        # print("mean value: ", np.mean(x))

        self.send_data1 = x.T.tolist()


        #print("filter: ",time.time() - current_time)
        current_time = time.time()
        
        ''' downsampling '''
        if self.send_ts and self.stream_params['downSampling']:
            for m in range(round(self.stream_params['chunkSize'] / self.stream_params['downSamplingFactor'])):
                end_idx = min((m + 1) * self.stream_params['downSamplingFactor'], len(self.send_data1))
                for ch_idx in range(len(self.send_data1[0])):
                    buf1 = [self.send_data1[n][ch_idx] for n in range(m * self.stream_params['downSamplingFactor'], end_idx)]
                    buf2 = [self.send_data2[n][ch_idx] for n in range(m * self.stream_params['downSamplingFactor'], end_idx)]
                    try:
                        self.stream_params['downSamplingBuffer1'][m][ch_idx] = sum(buf1) / len(buf1)
                        self.stream_params['downSamplingBuffer2'][m][ch_idx] = sum(buf2) / len(buf2)
                    except ZeroDivisionError:
                        print(m * self.stream_params['downSamplingFactor'])
                        print(end_idx)
                        print(m + 1)
                        print(self.stream_params['downSamplingFactor'])
                        print(len(self.send_data1))
                        print()
            self.send_data1 = self.stream_params['downSamplingBuffer1']
            self.send_data2 = self.stream_params['downSamplingBuffer2']

        current_time = time.time()

        self.tsBuffer = np.append(self.tsBuffer, timestamp)
        # print(self.dataBuffer.shape)
        # print(np.array(self.send_data1).shape)

        self.dataBuffer = np.append(self.dataBuffer, np.array(self.send_data1), axis = 0)
        if (self.dataBuffer.shape[0]) > self.stream_params['forwardSize']:

            self.dataBuffer = self.dataBuffer[-self.stream_params['forwardSize']:,:]
            self.tsBuffer = self.tsBuffer[-self.stream_params['forwardSize']:]

            x = self.dataBuffer.T # eeg_channels*512 

            x = x[:self.model_input_chan,:] # select first 8 channels


            x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)
            x = np2TT(x).to(device, dtype=torch.float)
            model.eval()
            x = model(x)
            x = x.view(self.model_input_chan, -1).detach().cpu().numpy()

            """
            x = x.reshape(1, self.model_input_chan, -1, 1) #resize x to make it (1, 8, 512, 1)
            x = model.predict(x[:, :, :self.stream_params['forwardSize'], :])
            x = x.reshape(self.model_input_chan, -1)
            """
            
            index = int( self.stream_params['chunkSize'] / self.stream_params['downSamplingFactor'] )
            self.send_data2 = x[:,-2*index:-1*index].T.tolist() 

        #print("model: ", time.time() - current_time)
        
        if any([self.send_ts]):
            ## self.sendData.emit(self.send_ts, self.send_data1, self.send_data2)
            self.update_stream(self.send_ts, self.send_data1, self.send_data2)      

    def update_stream(self,send_ts, send_data1, send_data2):
        # self.outlet_stream1.push_chunk(send_data1)
        # self.outlet_stream2.push_chunk(send_data2)
        x = np.asarray(send_data1).T
        print(x.min(axis=1), x.max(axis=1))
        self.outlet_stream1.push_chunk(send_data1, timestamp=send_ts[-1])
        self.outlet_stream2.push_chunk(send_data2, timestamp=send_ts[-1])
        

def main():
    stream = StreamerLSL()
    sample, timestamp = inlet.pull_chunk()
    stream.initialize()
    while True:
        stream.preprocess()
if __name__ == '__main__':
    main()
