import numpy as np

from qpsk import qpsk_encode

num_symbols = int(2e6)
samples_per_symbol = 2

# Generate data
bits = np.tile([1, 0, 1, 0, 1, 0, 1, 0], int((num_symbols * 2) / 8)) 

tx = qpsk_encode(bits, samples_per_symbol)
    
tx_flat = tx.view(np.float64) * 1024

#Write to file
ints = tx.astype('int16')
ints.tofile("samples.bin")
