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
parser.add_argument("-fd", "--freqdelay", type=int, default=1, help="data path")

param = parser.parse_args()

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
STFTmag, CIFpos, tremap = a_wav.retrieve_values()