import numpy as np
from rtlsdr import RtlSdr

from qpsk import qpsk_decode

samples_per_symbol = 2
num_symbols = 65536
num_samples = num_symbols * samples_per_symbol
symbol_rate = 500e3
sample_rate = symbol_rate * samples_per_symbol

sdr = RtlSdr()

# configure device
sdr.sample_rate = sample_rate  # Hz
sdr.center_freq = 920e6     # Hz
sdr.gain = 'auto'

rx = sdr.read_samples(num_samples)
rx = sdr.read_samples(num_samples)

qpsk_decode(rx, samples_per_symbol)

