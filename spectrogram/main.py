'''
Reassign spectrogram

Hyungwon Yang
NAMZ
18.10.15
'''

import os
import sys
import argparse

from src.reassign_spec import data_process

''' parser argument type.
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

# data and parameter
parser.add_argument("-D", "--data", type=str, default="a.wav", help="data path")

parser.add_argument("-wsi", "--win_size", type=int, default=0.025, help="window size")
parser.add_argument("-wst", "--win_step", type=int, default=0.01, help="window step")
parser.add_argument("-win", "--window", type=str, default='kaiser', help="window method")
parser.add_argument("-fn", "--fftn", type=int, default=1024, help="the number of fft bins")
parser.add_argument("-lc", "--low_cut", type=int, default=10, help="low frequency limit")
parser.add_argument("-hc", "--high_cut", type=int, default=5000, help="higt frequency limit")
parser.add_argument("-c", "--clip", type=int, default=-30, help="Db under the clip will not be plotted")
parser.add_argument("-d", "--delay", type=int, default=1, help="audio sample delay")
parser.add_argument("-fd", "--freqdelay", type=int, default=1, help="frequency delay")

# if you set this 'both' : time and frequency values will be reassigned.
# if you set this 'time' : only time value will be reassigned.
# if you set this 'frequency' : only frequency value will be reassigned.
# if you set this 'spectrogram' : not reassigned spectrogram.
parser.add_argument("-vt", "--value_type", type=str, default='both', help="values to be reassigned. both, time only, or frequency only.")

param = parser.parse_args()
'''
# input argument type: dictionary
param = {}
param['data'] = 'a.wav'
param['win_size'] = 0.025
param['win_step'] = 0.01
param['window'] = 'kaiser' # kaiser, rectangular, hamming, hanning, blackman
param['fftn'] = 1024
param['low_cut'] = 10
param['high_cut'] = 5000
param['clip'] = -30
param['delay'] = 1
param['freqdelay'] = 1
param['value_type'] = 'spectrogram'  # both, time, frequency, spectrogram


# import processing class.
a_wav = data_process(param)

# apply window
a_wav.apply_window()
# apply fft
a_wav.apply_fft()
# extract specific frequency.
a_wav.extract_postive()
# critical part: calculate different angles.
a_wav.compute_angles()
# normalization
a_wav.get_magnitude()
# plot reassign spectrogram
a_wav.plot_Rspectrogram()
# retrieve reassigned values.
# STFTmag, CIFpos, tremap = a_wav.retrieve_values()