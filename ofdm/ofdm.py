import numpy as np
import matplotlib.pyplot as plt

N    = 256
NCP  = 32
MPL  = 16
NSym = N + NCP

#Generate 2 random symbols
sym1 = np.random.randn(N) + 1j*np.random.randn(N)
sym2 = np.random.randn(N) + 1j*np.random.randn(N)

print("TX symbol 2")
#print(sym1[0:16])
print(sym2[0:16])

#OFDM modulate
sym1_fft = np.fft.ifft(sym1)
sym2_fft = np.fft.ifft(sym2)

#Prepend cyclic prefix
sym1_cp = np.concatenate((sym1_fft[-NCP:], sym1_fft))
sym2_cp = np.concatenate((sym2_fft[-NCP:], sym2_fft))

#Combined message consisting of 2 symbols
tx = np.concatenate((sym1_cp, sym2_cp))

#Channel response
h = np.random.randn(MPL) + 1j*np.random.randn(MPL)
rx = np.convolve(tx, h)
print("Channel response")
print(h)

#slice symbols
STO = -10 #Symbol timing offset
sym1_rx = rx[NCP+STO : N+NCP+STO]
sym2_rx = rx[NSym+NCP+STO : NSym+N+NCP+STO]

#Decode the symbols
sym1_dec = np.fft.fft(sym1_rx)
sym2_dec = np.fft.fft(sym2_rx)

#print("Decoded RX symbols")
#print(sym1_dec[0:16])
#print(sym2_dec[0:16])

print("Inferred channel response")
inferred_h = np.fft.ifft(sym1_dec / sym1) 
print(inferred_h[0:NCP + 2])

print("Inferred decoded sym2")
print(sym2_dec[0:16] * (sym1[0:16] / sym1_dec[0:16]))

