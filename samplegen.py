import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from commpy import filters

num_symbols = int(2e6)
samples_per_symbol = 2

# Generate data
bits = np.tile([1, 0, 1, 0, 1, 0, 1, 0], int((num_symbols * 2) / 8)) 

#Synchronous scrambler to make the data look random
scrambled = np.zeros(len(bits), dtype=bool)
for idx in range(len(bits)):
    scrambled[idx] = bits[idx] ^ scrambled[idx - 39] ^ scrambled[idx - 58]

# Upsample
x = np.zeros(num_symbols * samples_per_symbol, dtype=complex)
for idx, bitss in enumerate(scrambled.reshape((num_symbols, 2))):
    x[idx * samples_per_symbol] = (bitss[0]*2-1) + 1j*(bitss[1]*2-1)

# Generate root raised cosine
num_taps = 20 * samples_per_symbol
alpha = 0.35
t, h = filters.rrcosfilter(num_taps, alpha, 1, samples_per_symbol)

# Interpolate with RRC
tx = np.convolve(x, h)

tx_flat = tx.view(np.float64) * 1024

#print(tx_flat[400:500])

#Write to file
ints = tx_flat.astype('int16')
#print(ints[400:500])
ints.tofile("samples.bin")
