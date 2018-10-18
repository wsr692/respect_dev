'''
Reassign spectrogram

Hyungwon Yang
NAMZ
18.10.15
'''

import os
import sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d

class data_process(object):

    def __init__(self, parameters):
        print("Importing audio file and parameters.")
        self.param = parameters
        self.data = self.param['data']
        self.signal, self.Fs = sf.read(self.data)
        if self.signal.shape[0] == 0:
            print("Audio file is not imported.")
            exit()
        else:
            print("Audio file is imported successfully.")

        self.win_size = self.Fs * self.param['win_size']
        self.win_step = self.Fs * self.param['win_step']
        self.win_method = self.param['window']
        self.fftn = self.param['fftn']
        self.low_cut = self.param['low_cut']
        self.high_cut = self.param['high_cut']
        self.clip = self.param['clip']

        self.wave_len = len(self.signal)
        self.delay = self.param['delay']
        self.freqdelay = self.param['freqdelay']
        self.value_type = self.param['value_type']

    def windowing(self, shape='kaiser'):

        beta = 14
        if shape == 'rectangular':
            beta = 0
        elif shape == 'hamming':
            beta = 5
        elif shape == 'hanning':
            beta = 6
        elif shape == 'blackman':
            beta = 8.6
        elif shape == 'kaiser':
            beta = 14
        else:
            print('window method is not recognizable. kaiser method will be used.')
        window = np.kaiser(self.win_size, beta)
        return window

    def apply_window(self):

        print('windowing audio signals.')
        step = self.win_size - self.win_step
        self.offset = np.arange(0, self.wave_len - self.win_size - 1, step, dtype='int32')
        windowed = self.windowing(self.win_method)
        tmp_ww = np.array(windowed, ndmin=2, dtype='float32')
        WW = np.transpose(np.tile(tmp_ww, (len(self.offset),1)))

        tmp_idx = np.array(np.arange(len(windowed)), ndmin=2, dtype='int32')
        idx = np.transpose(np.tile(tmp_idx, (len(self.offset), 1)))
        for val in range(len(self.offset)):
            idx[:, val] += self.offset[val]

        self.S = np.multiply(self.signal[idx+1], WW)
        self.Sdel = np.multiply(self.signal[idx], WW)

    def apply_fft(self):
        print('applying fft.')
        self.STFT = np.fft.fft(self.S, self.fftn, axis=0)
        self.STFTdel = np.fft.fft(self.Sdel, self.fftn, axis=0)
        self.STFTfreqdel = np.concatenate((np.reshape(self.STFT[self.fftn-1,:],[1,len(self.offset)]),self.STFT[0:self.fftn-1,:]))

    def extract_postive(self):

        if np.remainder(self.fftn, 2) == 1:
            ret_n = np.round((self.fftn-1)/2)
        else:
            ret_n = self.fftn/2

        if self.high_cut > self.Fs * (ret_n-1)/self.fftn:
            self.high_cut = self.Fs * (ret_n-1)/self.fftn
        self.highindex = np.round(self.high_cut/self.Fs*self.fftn).astype('int32')

        self.lowindex = np.round(self.low_cut/self.Fs*self.fftn).astype('int32')
        if self.lowindex == 0:
            self.lowindex = 1

        self.STFTpos = self.STFT[self.lowindex:self.highindex+1,:]
        self.STFTdelpos = self.STFTdel[self.lowindex:self.highindex+1,:]
        self.STFTfreqdelpos = self.STFTfreqdel[self.lowindex:self.highindex+1,:]

    def compute_angles(self):
        print('reassigning the values.')
        C = self.STFTpos * np.conj(self.STFTdelpos)
        argC = np.mod(np.angle(C), 2*np.pi)
        # remapping frequency vectors.
        self.CIFpos = (self.Fs/self.delay) * argC / (2*np.pi)


        L = self.STFTpos * np.conj(self.STFTfreqdelpos)
        argL = np.mod(np.angle(L), -2*np.pi)
        # adjusting time axis.
        LGDpos = -((self.fftn/self.Fs) * argL) / (2*np.pi)
        self.t = (self.offset + self.win_size/2) / self.Fs
        self.tremap = np.zeros(LGDpos.shape)
        for val in range(0,self.highindex-self.lowindex+1):
            self.tremap[val,:] = LGDpos[val,:] + self.t - (self.win_size/2 - 1) / self.Fs

    def get_magnitude(self):

        STFTmag = np.abs(self.STFTpos)
        STFTmag = STFTmag / np.max(STFTmag)
        self.STFTmag = 20 * np.log10(STFTmag)

    def retrieve_values(self, type='both'):

        print('Value Information')
        print('1st: STFTmag')
        print('2nd: CIFpos')
        print('3rd: tremap')
        STFTplot = np.reshape(self.STFTmag, [self.STFTmag.shape[0] * self.STFTmag.shape[1]])
        CIFplot = np.reshape(self.CIFpos, [self.CIFpos.shape[0] * self.CIFpos.shape[1]])
        tremap = np.reshape(self.tremap, [self.tremap.shape[0] * self.tremap.shape[1]])
        if type == 'both':
            print('Both Time and Frequency Values are reassigned.')
        elif type == 'time':
            print('Only Time Value is reassigned.')
            time_len = self.CIFpos.shape[1]
            CIFplot = np.zeros(len(CIFplot))
            arcC = ((self.Fs/self.delay) * np.linspace(0, np.pi, self.CIFpos.shape[0])) / (2*np.pi)
            tmp_idx = 0
            for arc_idx in range(self.CIFpos.shape[0]):
                CIFplot[tmp_idx:tmp_idx+time_len] = np.tile(arcC[arc_idx],time_len)
                tmp_idx += time_len
        elif type == 'frequency':
            print('Only Frequency Value is reassigned.')
            tremap = np.tile(self.t, self.CIFpos.shape[0])
        elif type == 'spectrogram':
            print('Not reassigned simple spectrogram.')
            time_len = self.CIFpos.shape[1]
            CIFplot = np.zeros(len(CIFplot))
            arcC = ((self.Fs/self.delay) * np.linspace(0, np.pi, self.CIFpos.shape[0])) / (2*np.pi)
            tmp_idx = 0
            for arc_idx in range(self.CIFpos.shape[0]):
                CIFplot[tmp_idx:tmp_idx+time_len] = np.tile(arcC[arc_idx],time_len)
                tmp_idx += time_len
            tremap = np.tile(self.t, self.CIFpos.shape[0])

        plot_these = np.where(STFTplot >= self.clip) and np.where(self.low_cut <= CIFplot) and np.where(CIFplot <= self.high_cut) and \
                     np.where(self.t[0] <= tremap) and np.where(tremap <= self.t[-1])

        if len(STFTplot) != len(plot_these[0]):
            STFTplot = STFTplot[plot_these]
            CIFplot = CIFplot[plot_these]
            tremap = tremap[plot_these]

        return STFTplot, CIFplot, tremap

    def plot_Rspectrogram(self):
        print('plotting result.')
        STFTplot, CIFplot, tremap = self.retrieve_values(self.value_type)

        # f = self.Fs*np.arange(self.lowindex-1,self.highindex-1) / self.fftn

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(1000*tremap, CIFplot, STFTplot)
        plt.show()

