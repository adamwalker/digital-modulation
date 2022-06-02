import numpy as np
from commpy import impairments

from qpsk import qpsk_encode, qpsk_decode

samples_per_symbol = 2
num_symbols = 65536

bits = np.tile([1, 0, 1, 0, 1, 0, 1, 0], int((num_symbols * 2) / 8)) 

#Modulate
tx = qpsk_encode(bits, samples_per_symbol)

#Channel
rx = tx
#Add delay
pre = np.zeros(99, dtype=complex)
rx = np.concatenate((pre, rx))
#Add noise
n = (np.random.randn(len(rx)) + 1j*np.random.randn(len(rx)))/np.sqrt(2)
rx = rx + 0.1 * n
#Add frequency offset
rx = impairments.add_frequency_offset(rx, 1, 0.05)

#Demodulate
qpsk_decode(rx, samples_per_symbol)
