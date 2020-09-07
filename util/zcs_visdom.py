import os
import time
import numpy as np
from visdom import Visdom

class ZcsVisdom:
    def __init__(self, server='10.160.82.54', port=8899):
        self.vis = Visdom(server=server, port=port)
        assert self.vis.check_connection()

        self.wins = {}

    def plot(self, data, win):
        '''
        array: np.array
        win: str
        '''
        x = np.arange(data.shape[0])
        self.vis.line(data, x, win=win)#, update='append')

    def append(self, data, win, opts):
        '''
        data是个list，长度和绘制的折线数一致
        '''
        assert isinstance(data, list)
        if win not in self.wins.keys():
            self.vis.close(win)
            self.wins[win] = 0
        y = [data]
        x = [self.wins[win]]
        self.vis.line(Y=y, X=x, win=win, update='append', opts=opts)
        self.wins[win] += 1

if __name__ == '__main__':
    #from scipy.io import wavfile

    #samplerate, pcm = wavfile.read('./0002.cgmm.wav')

    zv = ZcsVisdom(server='10.160.82.54', port=8899)
    
    #y1 = np.ones_like(pcm) * 100
    #y2 = np.ones_like(pcm) * 1000
    #y = np.vstack([y1, y2])
    #print(y.shape)
    #zv.plot(y.T, 'wave')

    for i in range(10):
        zv.append([i*2, i*3], 'zcs', opts={'title':'test', 'legend':['loss1', 'loss2']})

    for i in range(60):
        zv.append([i], 'zcs2', opts={'title':'test2', 'legend':['loss1']})
