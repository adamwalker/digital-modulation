import numpy as np

from qpsk import qpsk_encode

num_symbols = int(2e6)
samples_per_symbol = 2

# Generate data
bits = np.tile([1, 0, 1, 0, 1, 0, 1, 0], int((num_symbols * 2) / 8)) 

qpsk_encode(bits, num_symbols, samples_per_symbol)
