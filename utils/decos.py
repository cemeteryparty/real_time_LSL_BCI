import sys
import os


class StreamInfo_():  # StreamInfo

    def __init__(self, stream):
        super(StreamInfo_, self).__init__()
        self.__streamInstance = stream
        self.name   = stream.name()
        self.type   = stream.type()
        self.srate  = stream.nominal_srate()
        self.n_chan = stream.channel_count()
        self.ssid   = stream.source_id()

    def lsl_stream(self):
        return self.__streamInstance

    def __str__(self):
        # return stream.as_xml()
        return "\n".join([
            " > Stream Name  : {}".format(self.name),
            " > Stream Type  : {}".format(self.type),
            " > # of Channel : {}".format(self.n_chan),
            " > Sampling Rate: {}".format(self.srate),
            " > Stream SrcID : {}".format(self.ssid)
        ])