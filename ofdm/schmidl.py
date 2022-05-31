import numpy as np
import matplotlib.pyplot as plt
import cmath as cm

N    = 256
L    = 128
NCP  = 32
NSym = N + NCP

#Generate random symbols
sym = np.random.randn(N) + 1j*np.random.randn(N)
sym[1::2] = 0;
#print(sym)

#OFDM modulate
sym_fft = np.fft.ifft(sym)

#Prepend cyclic prefix
sym_cp = np.concatenate((sym_fft[-NCP:], sym_fft))

#print(sym_cp)

#plt.figure(0)
#plt.plot(sym_cp)
#plt.show()

#Channel
rx = sym_cp

rx_delayed = np.concatenate((np.zeros(L, dtype = complex), rx))
rx_padded  = np.concatenate((rx, np.zeros(L, dtype = complex)))

conj = rx_padded * np.conj(rx_delayed)

conj_delayed = np.concatenate((np.zeros(L, dtype = complex), conj))
conj_padded  = np.concatenate((conj, np.zeros(L, dtype = complex)))

res = np.cumsum(conj_padded - conj_delayed)

plt.figure(0)
plt.plot(abs(res))
plt.show()

delay = np.argmax(abs(res))
print(delay)
print(cm.phase(res[delay]))
